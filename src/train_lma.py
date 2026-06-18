"""
Window-level dance-style classification on faithful per-window LMA features.

CORRECTED PIPELINE:
  * One sample = one 55-frame WINDOW (a faithful (1,61) LMA descriptor), NOT one frame.
    The input dir holds one `<clip>_features.npy` of shape (num_windows, 61) per clip.
  * CV groups = clip id, so windows of a clip never straddle train/test.
    `--mode original` (GroupKFold, the DEFAULT) is the honest generalization split.
    `--mode shuffled` (plain KFold) ignores clips -> same-clip windows leak across
    folds -> inflated; kept ONLY as a leaky diagnostic baseline.
  * Reports BOTH window-level accuracy and clip-level accuracy (majority vote over a
    clip's windows) -- the latter is the real "does this whole motion belong to genre X".
"""
import os
import glob
import re
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
import cupy as cp
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm

# Mapping from AIST++ filename codes to full Genre names [cite: 73]
GENRE_MAP = {
    'BR': 'Break', 'HO': 'House', 'JB': 'Jazz Ballet', 'JS': 'Jazz Street',
    'KR': 'Krump', 'LH': 'LA Hip Hop', 'LO': 'Lock', 'MH': 'Middle Hip Hop',
    'PO': 'Pop', 'WA': 'Waack'
}

def _dict_to_matrix(d):
    """
    Convert a dictionary of 1D arrays (same length) to a (T, F) matrix.
    If values are already arrays of shape (T,), they become columns.
    """
    keys = sorted(d.keys())
    cols = []
    # Filter keys that are arrays and share the same max length
    lengths = {k: (np.asarray(d[k]).shape[0] if hasattr(d[k], '__len__') else 0) for k in keys}
    max_len = max(lengths.values()) if lengths else 0
    keys = [k for k in keys if lengths[k] == max_len and max_len > 0]

    for k in keys:
        arr = np.asarray(d[k])
        if arr.ndim == 1:
            cols.append(arr)
        elif arr.ndim == 2 and arr.shape[1] == 1:
            cols.append(arr[:, 0])
        else:
            # flatten multi-column entries into separate columns
            for i in range(arr.shape[1]):
                cols.append(arr[:, i])
    if not cols:
        raise ValueError("No valid feature arrays found in dict.")
    return np.stack(cols, axis=1)


def _clip_majority_vote(groups_test, y_true, y_pred):
    """Aggregate window-level predictions to ONE prediction per clip (majority vote).
    y_true is constant within a clip. Returns (clip_accuracy, n_clips)."""
    from collections import Counter
    preds, truth = {}, {}
    for g, t, p in zip(groups_test, y_true, y_pred):
        preds.setdefault(g, []).append(p)
        truth[g] = t
    correct = sum(Counter(ps).most_common(1)[0][0] == truth[g]
                  for g, ps in preds.items())
    return correct / len(preds), len(preds)

def load_dataset(input_dir):
    """
    Robust loading of features.
    - Handles .npy dictionaries or arrays.
    - Filters by AIST++ genre codes (gBR, gHO, etc.) in filenames.
    """
    X_list = []
    y_list = []
    groups_list = []

    search_path = os.path.join(input_dir, "*.npy")
    all_npy_files = sorted(glob.glob(search_path))

    if not all_npy_files:
        print(f"[!] No .npy files found in {input_dir}")
        return None, None, None

    print(f"Scanning {len(all_npy_files)} files...")

    # Regex to find 'gXX' where XX is a known genre code
    genre_codes = "|".join(GENRE_MAP.keys())
    genre_regex = re.compile(rf"g({genre_codes})", re.IGNORECASE)

    for f_path in tqdm(all_npy_files, desc="Processing Files", unit="file"):
        filename = os.path.basename(f_path)

        # strict filter for feature files only (case-insensitive)
        if not filename.lower().endswith("_features.npy"):
            continue

        # Detect Genre
        match = genre_regex.search(filename)
        if not match:
            print(f"  [?] Skipping (no genre token): {filename}")
            continue

        style_code = match.group(1).upper()
        if style_code not in GENRE_MAP:
            print(f"  [?] Unknown genre code in filename: {filename}")
            continue
        label = GENRE_MAP[style_code]

        try:
            raw = np.load(f_path, allow_pickle=True)

            # unwrap 0-d object arrays that contain a dict
            if isinstance(raw, np.ndarray) and raw.dtype == object and raw.ndim == 0:
                raw = raw.item()

            if isinstance(raw, dict):
                data = _dict_to_matrix(raw)
            else:
                data = np.asarray(raw)

            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.ndim != 2:
                print(f"  [!] Unsupported data shape for {filename}: {data.shape}. Skipping.")
                continue

            if data.shape[1] != 61:
                print(f"  [!] Warning: {filename} has {data.shape[1]} features (expected 61). Proceeding anyway.")

            video_id = filename.replace('_features.npy', '')

            X_list.append(data)
            y_list.append(np.array([label] * data.shape[0], dtype=object))
            groups_list.append(np.array([video_id] * data.shape[0], dtype=object))

        except Exception as e:
            print(f"  [!] Error reading {filename}: {e}")

    if not X_list:
        print("[!] No valid data loaded.")
        return None, None, None

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)

    print(f"Loaded {X.shape[0]} window samples from {len(np.unique(groups))} clips.")
    print(f"Feature dimension: {X.shape[1]}")
    return X, y, groups

def train_and_evaluate(X, y, groups, mode, save_model_path=None):
    # 1. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y).astype(np.int32)
    print(f"Labels encoded. Classes: {le.classes_}")

    # 2. Move to GPU immediately
    # float32 is significantly faster on GPUs than float64
    X_gpu = cp.array(X, dtype=cp.float32)
    y_gpu = cp.array(y_encoded, dtype=cp.int32)
    
    # We keep groups on CPU because GroupKFold needs them to calculate indices
    groups = np.asarray(groups)
    fold_accuracies = []
    clip_accuracies = []

    print("\n" + "="*60)
    print("Starting GPU-Accelerated Evaluation")
    print("="*60)

    # Inner GridSearch using cuML's version
    if mode == 'original':
        outer_cv = GroupKFold(n_splits=3)
        inner_cv = GroupKFold(n_splits=3)
    elif mode == 'shuffled':
        print("  [!] WARNING: 'shuffled' is a LEAKY diagnostic baseline — windows of the "
              "same clip can land in both train and test, inflating accuracy. "
              "Do NOT report it as a generalization number.")
        outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

    # Use the encoded CPU y and groups to split
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_gpu, y_gpu, groups)):
        print(f"\nProcessing Fold {fold + 1}/3...")

        # Slice GPU arrays directly to avoid host copies
        X_train_gpu, X_test_gpu = X_gpu[train_idx], X_gpu[test_idx]
        y_train_gpu, y_test_gpu = y_gpu[train_idx], y_gpu[test_idx]
        groups_train = groups[train_idx]

        # cuML RF is highly parallelized
        rf = RandomForestClassifier(random_state=42)
        
        # Massive estimator counts to really push the VRAM
        param_grid = {
            'n_estimators': [300, 500],
            'max_depth': [20, 30],
            'min_samples_split': [2, 5]
        }

        grid = GridSearchCV(rf, param_grid, cv=inner_cv, verbose=1)
        
        n_candidates = len(param_grid['n_estimators']) * \
               len(param_grid['max_depth']) * \
               len(param_grid['min_samples_split'])
        total_fits = n_candidates * 3  # 3 is the inner_cv n_splits

        print(f"  > Grid Search: Training {n_candidates} candidates for a total of {total_fits} fits...")

        if mode == 'original':
            grid.fit(X_train_gpu, y_train_gpu, groups=groups_train)
        elif mode == 'shuffled':
            grid.fit(X_train_gpu, y_train_gpu)

        best_model = grid.best_estimator_
        print(f"  > Best Params: {grid.best_params_}")

        # Predict stays on GPU
        y_pred_gpu = best_model.predict(X_test_gpu)
        
        # Convert back to CPU only for metrics/reporting
        y_pred = cp.asnumpy(y_pred_gpu)
        y_test_cpu = cp.asnumpy(y_test_gpu)
        
        acc = accuracy_score(y_test_cpu, y_pred)
        fold_accuracies.append(acc)

        # Clip-level (sequence) accuracy: majority-vote each clip's window predictions.
        # In 'original' (grouped) mode the test clips are fully held out, so this is the
        # honest "does this whole motion belong to genre X" number.
        groups_test = groups[test_idx]
        clip_acc, n_clips = _clip_majority_vote(groups_test, y_test_cpu, y_pred)
        clip_accuracies.append(clip_acc)

        print(f"  > Fold window accuracy: {acc:.4f}   |   "
              f"clip-level (majority vote): {clip_acc:.4f} over {n_clips} clips")
        print("  > Window-level Classification Report:")
        print(classification_report(y_test_cpu, y_pred, target_names=le.classes_, digits=4, zero_division=0))

        if save_model_path:
            fname = os.path.join(save_model_path, f"best_model_fold{fold+1}.joblib")
            # cuML models can be saved via joblib; pickle is also common
            joblib.dump(best_model, fname)
            print(f"  > Saved model to: {fname}")

    print("\n" + "="*60)
    print(f"FINAL MEAN WINDOW ACCURACY: {np.mean(fold_accuracies):.4f}")
    print(f"FINAL MEAN CLIP ACCURACY (majority vote): {np.mean(clip_accuracies):.4f}")
    split_desc = ("GroupKFold by clip — honest generalization" if mode == 'original'
                  else "plain KFold — LEAKY baseline (same-clip windows straddle folds)")
    print(f"  Split: {split_desc}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_models", type=str, default=None,
                        help="Optional folder to save best models per fold")
    parser.add_argument("--mode", type=str, choices=['original', 'shuffled'],
                        default='original',
                        help="'original' = GroupKFold by clip (honest, DEFAULT). "
                             "'shuffled' = plain KFold ignoring clips (LEAKY baseline; "
                             "same-clip windows straddle folds -> inflated).")
    args = parser.parse_args()

    X, y, groups = load_dataset(args.data_dir)
    if X is not None:
        if args.save_models:
            os.makedirs(args.save_models, exist_ok=True)
        train_and_evaluate(X, y, groups, mode=args.mode, save_model_path=args.save_models)