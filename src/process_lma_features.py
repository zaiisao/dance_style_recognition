import torch
import torchvision
import numpy as np
import argparse
import cv2
import os
import glob

from floor import MoGeFloorEstimator, FlatFloorEstimator, FloorEstimator  # noqa: F401
from pose import NLFPoseEstimator, PoseEstimator  # noqa: F401
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.lma_extractor import LMAExtractor
from utils.visualizer import render_comprehensive_dashboard

# IdentityFloor + compute_lma_descriptor live in the lightweight, dependency-free
# lma_descriptor module so consumers (e.g. the WHAM suggestive-motion pipeline) can
# import the data-producing extractor without pulling in torch/cv2/MoGe.
from lma_descriptor import IdentityFloor, compute_lma_descriptor, hull_volume

# Per-frame pose now lives in pose.py as the pluggable NLFPoseEstimator (default).
# process_single_video takes a `pose_estimator` so the pose backend can be overridden.
# See pose.py.

# Floor estimation now lives in floor.py as the pluggable MoGeFloorEstimator (default)
# / FlatFloorEstimator. process_single_video takes a `floor_estimator` so the depth model
# can be overridden or skipped entirely. See floor.py.

def verify_pipeline_integrity(all_joints, all_volumes, all_floor_models):
    """
    Analyzes the captured data for physical consistency.
    """
    print("\n" + "="*40)
    print("      PIPELINE INTEGRITY REPORT      ")
    print("="*40)

    # 1. Detection Rate
    total_frames = len(all_joints)
    # Check for valid numpy arrays (not empty lists)
    valid_frames = [j for j in all_joints if len(j) > 0] 
    
    detection_rate = (len(valid_frames) / total_frames) * 100 if total_frames > 0 else 0
    
    print(f"[-] Detection Stability:")
    print(f"    Total Frames: {total_frames}")
    print(f"    Valid Detections: {len(valid_frames)} ({detection_rate:.1f}%)")
    
    if len(valid_frames) == 0:
        print("[!] CRITICAL FAILURE: No humans detected in any frame.")
        return

    # 2. Geometric Grounding (Pelvis Height)
    pelvis_heights = []
    for i, j in enumerate(valid_frames):
        if len(j) == 0: continue

        # j is (24, 3), so j[0] is the Pelvis vector
        pelvis_pos = j[0]
        current_model = all_floor_models[i]
        
        # Predict Floor Y (Height) using Pelvis Z (Depth)
        floor_y = current_model.predict(pelvis_pos[2].reshape(-1, 1))[0]
        
        # Height = Floor Y (Bottom) - Pelvis Y (Top)
        h = floor_y - pelvis_pos[1]
        pelvis_heights.append(h)

    pelvis_heights = np.array(pelvis_heights)
    mean_h = np.mean(pelvis_heights)
    std_h = np.std(pelvis_heights)

    print(f"\n[-] Geometric Grounding (Pelvis Height):")
    print(f"    Mean Height: {mean_h:.3f} m (Target: ~0.85m - 1.0m)")
    print(f"    Std Dev:     {std_h:.3f} m")
    
    if mean_h < 0.5 or mean_h > 1.3:
        print("    [!] WARNING: Dancer scale/floor estimation seems off.")
    else:
        print("    [OK] Scale looks realistic.")

    # 3. Volumetric Consistency (Shape Component)
    volumes = np.array(all_volumes)
    valid_vols = volumes[volumes > 0.00001]
    
    print(f"\n[-] Volumetric Consistency (Shape):")
    if len(valid_vols) > 0:
        mean_v = np.mean(valid_vols)
        print(f"    Mean Volume: {mean_v:.4f} m^3 (Target: ~0.06 - 0.09)")
        
        # Simple ASCII Plot
        print("\n    Volume Trend (last 50 frames):")
        if np.max(valid_vols) > 0:
            normalization = 20 / np.max(valid_vols)
            for v in valid_vols[-50:]:
                bar = "#" * int(v * normalization)
                print(f"    |{bar}")
    else:
        print("    [!] CRITICAL: No valid volumes calculated.")

def verify_lma_integrity(npy_path, plot_output_path="lma_verification_plot.png"):
    """
    Comprehensive audit for the 61-feature LMA descriptor.
    Validates components: Body (12), Effort (28), Space (8), Shape (1), Kinematics (6).
    """
    print(f"\n{'='*60}\nAUDITING LMA FEATURE VECTOR: {npy_path}\n{'='*60}")
    
    try:
        # Handling the wrap from np.save(..., allow_pickle=True)
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"[!] FATAL: Loading failed. Error: {e}")
        return

    keys = list(data.keys())
    n_features = len(keys)
    n_frames = len(data[keys[0]])
    
    # Requirement: Vector must contain exactly 61 features 
    print(f"[-] Descriptor Structure:")
    print(f"    Total Features: {n_features} (Target: 61)")
    print(f"    Total Frames:   {n_frames}")
    
    if n_features != 61:
        print(f"    [!] WARNING: Feature count mismatch! Found {n_features}, expected 61.")

    # ---------------------------------------------------------
    # 1. EFFORT COMPONENT VALIDATION (28 Features)
    # ---------------------------------------------------------
    # Effort captures intention/energy across Space, Weight, Time, and Flow.
    print(f"\n[-] Component 1: Effort (Energy & Dynamics)")
    
    # Check Weight (Kinetic Energy) - Eq 4
    weight_globals = data.get('Effort_Weight_Global', np.zeros(n_frames))
    if np.max(weight_globals) > 500: # Threshold for standard human movement in m/s^2
        print(f"    [!] FAIL: Weight values ({np.max(weight_globals):.2f}) suggest mm units.")
    else:
        print(f"    [OK] Weight (KE) scaling looks correct.")

    # Check Time (Acceleration) - Eq 5
    time_globals = data.get('Effort_Time_Global', np.zeros(n_frames))
    print(f"    [OK] Time (Acceleration) mean: {np.mean(time_globals):.3f} m/s^2")

    # ---------------------------------------------------------
    # 2. SPACE COMPONENT VALIDATION (8 Features)
    # ---------------------------------------------------------
    # Space describes relationship with kinesphere/personal space.
    print(f"\n[-] Component 2: Space (Kinesphere & Trajectory)")
    
    # Check Curvature: Path_Length / Displacement
    curvature = data.get('Traj_Curvature', np.zeros(n_frames))
    if np.any(curvature < -1e-6):
        print(f"    [!] FAIL: Negative curvature found. Check path/displacement computation.")
    else:
        print(f"    [OK] Curvature is non-negative (Min: {np.min(curvature):.3f}).")

    # ---------------------------------------------------------
    # 3. BODY COMPONENT & INITIATION (12 Features)
    # ---------------------------------------------------------
    # Body focuses on mechanics and initiation detection.
    print(f"\n[-] Component 3: Body (Initiation Triggers)")
    
    init_keys = [k for k in keys if "Initiation" in k]
    total_initiations = sum([np.sum(data[k]) for k in init_keys])
    if total_initiations == 0:
        print(f"    [!] WARNING: No initiation events detected. Threshold epsilon might be too high.")
    else:
        print(f"    [OK] Detected {int(total_initiations)} movement initiation events.")

    # ---------------------------------------------------------
    # 4. TEMPORAL EVOLUTION CHECK (SHAP Influence)
    # ---------------------------------------------------------
    # Temporal context significantly improves recognition performance.
    print(f"\n[-] Component 4: Temporal Context Audit")
    
    # Identify "Dead" features that don't change over time
    dead_features = [k for k in keys if np.std(data[k]) < 1e-6]
    if dead_features:
        print(f"    [!] WARNING: {len(dead_features)} features are static (Zero Variance).")
        print(f"        First 3 static: {dead_features[:3]}")
    else:
        print(f"    [OK] All 61 features show temporal evolution.")

    # ---------------------------------------------------------
    # 5. VISUALIZATION: Turab-Style Plotting
    # ---------------------------------------------------------
    # Style-specific representation improves with sliding window.
    print(f"\n[5] Generating Visualization: '{plot_output_path}'...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top Plot: Effort Components (Major predictors in SHAP plots)
    effort_plot_keys = ['Effort_Weight_Global', 'Effort_Time_Global', 'body_volume']
    for k in effort_plot_keys:
        if k in data:
            # Z-score for visual comparison
            norm = (data[k] - np.mean(data[k])) / (np.std(data[k]) + 1e-6)
            axes[0].plot(norm, label=k, alpha=0.8)
    
    axes[0].set_title("Primary Recognition Features (Normalized Evolution)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Bottom Plot: Initiation Events (Binary spikes)
    for k in init_keys[:3]: # Plot first 3 for clarity
        axes[1].step(range(n_frames), data[k], label=k, where='post')
    
    axes[1].set_title("Movement Initiation Triggers (Boolean Detection)")
    axes[1].set_xlabel("Frame Index")
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(plot_output_path)
    print(f"    [DONE] Verification complete.")

def process_single_video(
    video_path,
    output_dir,
    nlf_model=None,
    moge_model=None,
    device="cuda",
    viz=False,
    short_window=5,
    apply_smoothing=False,
    floor_estimator=None,
    pose_estimator=None,
):
    # Pose and floor are both pluggable.
    #  - pose_estimator: default NLF (the dance paper's per-frame pose). Override with your
    #    own backend (anything with estimate(frame) -> (joints3d, vertices3d)).
    #  - floor_estimator: default MoGe (the dance paper's depth-based floor). Pass
    #    FlatFloorEstimator() for ground-aligned joints (no depth model), or any object with
    #    estimate(frame) -> floor_model.
    # nlf_model / moge_model (preloaded models) are kept for backwards compatibility and are
    # wrapped here when no estimator is given.
    if pose_estimator is None:
        pose_estimator = NLFPoseEstimator(model=nlf_model, device=device)
    if floor_estimator is None:
        floor_estimator = MoGeFloorEstimator(model=moge_model, device=device)

    # Create dynamic filenames based on the specific video name
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nProcessing: {base_name}")

    # Define unique output paths for this specific video
    npy_output_path = os.path.join(output_dir, f"{base_name}_features.npy")
    video_output_path = os.path.join(output_dir, f"{base_name}_debug.mp4")
    plot_output_path = os.path.join(output_dir, f"{base_name}_plot.png")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    all_joints = []
    all_volumes = []
    all_vertices = []
    all_floor_models = []
    pelvis_depths = []
    pelvis_y_vals = []
    scene_cloud = None

    last_valid_volume = 0.0

    # (A debug shortcut that replayed a captured debug_data.npz fixture was removed for
    #  the public release; this is the real per-frame floor + pose fitting path.)
    current_floor_model = None
    with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # --- STAGE B: Floor Estimation (pluggable; default = MoGe) ---
            if frame_idx == 0:
                current_floor_model = floor_estimator.estimate(frame)

            # Keep frame-aligned floor models for downstream feature extraction.
            all_floor_models.append(current_floor_model)

            # --- STAGE A: Pose Estimation (Every Frame) ---
            # NLF must run every frame to capture the dance.
            joints3d, vertices3d = pose_estimator.estimate(frame)
                
            joints_np = None
            current_vol = last_valid_volume
                
            if len(vertices3d) > 0 and len(vertices3d[0]) > 0:
                # 1. Fetch from GPU ONCE
                verts_np = vertices3d[0].detach().cpu().numpy()
                joints_np = joints3d[0].detach().cpu().numpy()
                    
                # 2. Handle dimensions
                if verts_np.ndim == 3: verts_np = verts_np[0]
                if joints_np.ndim == 3: joints_np = joints_np[0]
                    
                # 3. Apply Scaling to the CPU copy (mm -> meters)
                # This ensures both the visualizer AND the volume calc get the scaled data
                if np.mean(np.abs(verts_np[:, 2])) > 10.0: 
                    verts_np /= 1000.0
                    joints_np /= 1000.0

                # 4. Save for Viz
                if viz:
                    all_vertices.append(verts_np.astype(np.float16))
                else:
                    all_vertices.append(None)

                # 5. Body volume (Shape feature; single definition in lma_descriptor).
                # Carries forward the last valid volume on a degenerate/<4-point frame.
                current_vol = hull_volume(verts_np, fallback=last_valid_volume)
                last_valid_volume = current_vol
            else:
                if viz:
                    all_vertices.append(None)
                
            if joints_np is not None:
                all_joints.append(joints_np)
            else:
                all_joints.append([]) # Keep list length consistent
                
            all_volumes.append(current_vol)

            if joints_np is not None:
                pelvis_depths.append(joints_np[0, 2]) # Pelvis Z
                pelvis_y_vals.append(joints_np[0, 1])  # Pelvis Y
                pbar.set_postfix(vol=f"{current_vol:.3f}")
            else:
                # Keep the lists the same length as all_joints
                pelvis_depths.append(np.nan)
                pelvis_y_vals.append(np.nan)
                
            frame_idx += 1
            pbar.update(1)
                
    cap.release()

    if len(pelvis_depths) > 0:
        z_array = np.array(pelvis_depths).reshape(-1, 1)
        # Filter frames where detection was successful
        valid_mask = ~np.isnan(pelvis_depths)
            
        if np.any(valid_mask):
            # One single call for the entire video
            all_floor_ys = current_floor_model.predict(z_array[valid_mask])
            actual_pelvis_ys = np.array(pelvis_y_vals)[valid_mask]
                
            # Calculate all heights at once
            all_heights = all_floor_ys - actual_pelvis_ys
            print(f"[-] Mean grounding height: {np.mean(all_heights):.3f}m")

    print("Video processing complete.")

    verify_pipeline_integrity(all_joints, all_volumes, all_floor_models)

    lma_dict, lma_matrix = compute_lma_descriptor(
        all_joints, all_volumes,
        all_floor_models,
        fps,
        window_size=55,
        short_window=short_window,
        apply_smoothing=apply_smoothing,
    )

    print(f"[-] Feature Extraction Complete")
    print(f"    Feature Matrix Shape: {lma_matrix.shape} (Frames x Features)")

    # 4. Save both formats
    np.save(npy_output_path, lma_matrix) 
    dict_output_path = npy_output_path.replace("_features.npy", "_dict.npy")
    np.save(dict_output_path, lma_dict) 

    print(f"    Saved Matrix to:   {npy_output_path}")
    print(f"    Saved Dictionary to: {dict_output_path}")
    
    verify_lma_integrity(dict_output_path, plot_output_path=plot_output_path)

    if viz:
        print("\n--- GENERATING VISUAL DEBUG ASSETS ---")
        render_comprehensive_dashboard(
            video_path, 
            all_joints, 
            all_vertices, 
            all_floor_models, 
            scene_cloud,
            lma_features=lma_dict,
            output_path=video_output_path
        )


def main():
    # 1. SETUP ARGUMENTS
    parser = argparse.ArgumentParser(description="Batch LMA Extraction")
    
    # Changed from 'input_dir' to 'input_path' to be more accurate
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to a single .mp4 file OR a folder of files")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Folder to save results")
    
    parser.add_argument("--viz", action="store_true", 
                        help="Enable debug video generation")

    parser.add_argument(
        "--short_window",
        type=int,
        default=5,
        help="Short lag window (frames) used for initiation and lagged-space metrics.",
    )
    parser.add_argument(
        "--apply_smoothing",
        action="store_true",
        help="Enable Savitzky-Golay smoothing.",
    )

    args = parser.parse_args()

    short_window = max(1, int(args.short_window))
    apply_smoothing = bool(args.apply_smoothing)

    print(
        f"Extractor config -> short_window={short_window}, apply_smoothing={apply_smoothing}"
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. DETERMINE INPUT TYPE
    video_files = []
    if os.path.isfile(args.input_path):
        # User provided a single file
        video_files = [args.input_path]
    elif os.path.isdir(args.input_path):
        # User provided a folder
        video_files = glob.glob(os.path.join(args.input_path, "*.mp4"))
    else:
        print(f"Error: {args.input_path} is not a valid file or directory.")
        return

    print(f"Found {len(video_files)} items to process.")

    # 3. SET UP BACKENDS (default = the dance pipeline's NLF pose + MoGe floor).
    # Both load their checkpoints lazily on the first frame. Swap either for a custom
    # estimator (e.g. FlatFloorEstimator() for ground-aligned joints) to customize.
    print("Setting up pose / floor backends...")
    pose_estimator = NLFPoseEstimator(device=device)
    floor_estimator = MoGeFloorEstimator(device=device)

    # 4. RUN SEQUENTIALLY
    # (If run by external MP script, this list will just contain 1 item)
    for video_path in video_files:
        try:
            process_single_video(
                video_path,
                args.output_dir,
                device=device,
                viz=args.viz,
                short_window=short_window,
                apply_smoothing=apply_smoothing,
                pose_estimator=pose_estimator,
                floor_estimator=floor_estimator,
            )
        except Exception as e:
            print(f"Failed on {video_path}: {e}")

if __name__ == "__main__":
    main()