import torch
import torchvision
import numpy as np
import argparse
import cv2
import os
import glob
from sklearn.linear_model import QuantileRegressor
from scipy.spatial import ConvexHull
from moge.model.v2 import MoGeModel
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.lma_extractor import LMAExtractor
from utils.visualizer import render_comprehensive_dashboard

def stage_a_nlf_implementation(frame, model, device="cuda"):
    # Convert BGR to RGB and move to GPU in one pipeline
    frame_tensor = torch.from_numpy(frame[..., ::-1].copy()).to(device)
    frame_tensor = frame_tensor.permute(2, 0, 1).float() / 255.0
    frame_batch = frame_tensor.unsqueeze(0)

    with torch.inference_mode():
        pred = model.detect_smpl_batched(frame_batch)
    
    return pred['joints3d'], pred['vertices3d']

def stage_b_floor_estimation(frame, model, device="cuda"):
    # 2. Process Image
    # Replaces: img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = torch.tensor(img / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    
    # 3. Infer 3D Point Map
    output = model.infer(input_tensor)
    points = output["points"].cpu().numpy() # Metric scale (H, W, 3)
    mask = output["mask"].cpu().numpy().astype(bool)
    valid_points = points[mask]

    target_count = 2000
    if len(valid_points) > target_count:
        step = len(valid_points) // target_count
        # [::step] is deterministic. It picks the same pixels every time.
        scene_cloud = valid_points[::step]
    else:
        scene_cloud = valid_points

    # [cite_start]4. Explicit Floor Fitting (Rejects the "lowest ankle" assumption [cite: 83])
    # Project to XZ-plane (x: right, z: forward depth in OpenCV camera coords)
    z = scene_cloud[:, 2].reshape(-1, 1)
    y = scene_cloud[:, 1]

    # [cite_start]Fit a line to the bottom 5% of points to handle tilt/slope [cite: 86]
    qr = QuantileRegressor(quantile=0.95, alpha=0, solver='highs')
    qr.fit(z, y)

    return qr, valid_points, scene_cloud

def verify_pipeline_integrity(all_joints, all_volumes, floor_model):
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
    for j in valid_frames:
        # j is (24, 3), so j[0] is the Pelvis vector
        pelvis_pos = j[0] 
        
        # Predict Floor Y (Height) using Pelvis Z (Depth)
        floor_y = floor_model.predict(pelvis_pos[2].reshape(-1, 1))[0]
        
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
    Comprehensive audit for the 55-feature LMA descriptor.
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
    
    # Requirement: Vector must contain exactly 55 features 
    print(f"[-] Descriptor Structure:")
    print(f"    Total Features: {n_features} (Target: 55)")
    print(f"    Total Frames:   {n_frames}")
    
    if n_features != 55:
        print(f"    [!] WARNING: Feature count mismatch! Found {n_features}, expected 55.")

    # ---------------------------------------------------------
    # 1. EFFORT COMPONENT VALIDATION (28 Features)
    # ---------------------------------------------------------
    # Effort captures intention/energy across Space, Weight, Time, and Flow[cite: 102].
    print(f"\n[-] Component 1: Effort (Energy & Dynamics)")
    
    # Check Weight (Kinetic Energy) - Eq 4 [cite: 112]
    weight_globals = data.get('Effort_Weight_Global', np.zeros(n_frames))
    if np.max(weight_globals) > 500: # Threshold for standard human movement in m/s^2
        print(f"    [!] FAIL: Weight values ({np.max(weight_globals):.2f}) suggest mm units.")
    else:
        print(f"    [OK] Weight (KE) scaling looks correct.")

    # Check Time (Acceleration) - Eq 5 [cite: 116]
    time_globals = data.get('Effort_Time_Global', np.zeros(n_frames))
    print(f"    [OK] Time (Acceleration) mean: {np.mean(time_globals):.3f} m/s^2")

    # ---------------------------------------------------------
    # 2. SPACE COMPONENT VALIDATION (8 Features)
    # ---------------------------------------------------------
    # Space describes relationship with kinesphere/personal space[cite: 119].
    print(f"\n[-] Component 2: Space (Kinesphere & Trajectory)")
    
    # Check Curvature: Path_Length / Displacement [cite: 118, 119]
    curvature = data.get('Traj_Curvature', np.ones(n_frames))
    if np.any(curvature < 0.99):
        print(f"    [!] FAIL: Curvature < 1.0 found. Check Triangle Inequality logic.")
    else:
        print(f"    [OK] Curvature logic verified (Min: {np.min(curvature):.3f}).")

    # ---------------------------------------------------------
    # 3. BODY COMPONENT & INITIATION (12 Features)
    # ---------------------------------------------------------
    # Body focuses on mechanics and initiation detection[cite: 93, 95].
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
    # Temporal context significantly improves recognition performance[cite: 13].
    print(f"\n[-] Component 4: Temporal Context Audit")
    
    # Identify "Dead" features that don't change over time
    dead_features = [k for k in keys if np.std(data[k]) < 1e-6]
    if dead_features:
        print(f"    [!] WARNING: {len(dead_features)} features are static (Zero Variance).")
        print(f"        First 3 static: {dead_features[:3]}")
    else:
        print(f"    [OK] All 55 features show temporal evolution.")

    # ---------------------------------------------------------
    # 5. VISUALIZATION: Turab-Style Plotting
    # ---------------------------------------------------------
    # Style-specific representation improves with sliding window[cite: 227].
    print(f"\n[5] Generating Visualization: '{plot_output_path}'...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top Plot: Effort Components (Major predictors in SHAP plots [cite: 172])
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

def process_single_video(video_path, output_dir, nlf_model, moge_model, device="cuda", viz=False):
    # Create dynamic filenames based on the specific video name
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nProcessing: {base_name}")

    # Define unique output paths for this specific video
    npy_output_path = os.path.join(output_dir, f"{base_name}_features.npy")
    video_output_path = os.path.join(output_dir, f"{base_name}_debug.mp4")
    plot_output_path = os.path.join(output_dir, f"{base_name}_plot.png")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    DEBUG_DURATION = 3.0 
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

    should_use_debug_data = False
    if should_use_debug_data:
        data = np.load("debug_data.npz", allow_pickle=True)
        all_joints = data['joints']
        all_volumes = data['volumes']
        floor_params = data['floor'] # Now you have the (N, 2) array of slope/intercept

        def recreate_floor_model(slope, intercept):
            # 1. Create a raw, untrained model
            model = QuantileRegressor()
            
            # 2. Manually inject the "learned" attributes
            # Note: Scikit-learn expects coef_ to be an array, even for 1D
            model.coef_ = np.array([slope]) 
            model.intercept_ = intercept
            
            # 3. (Optional) Trick Scikit-learn's validation checks
            # Some versions check if 'n_features_in_' exists to confirm fitting
            model.n_features_in_ = 1 
            
            return model

        # Usage Example with your loaded data:
        slope, intercept = floor_params[0] # Get frame 0
        floor_model = recreate_floor_model(slope, intercept)
    else:
        current_floor_model = None
        with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # --- STAGE B: Floor Estimation ---
                if frame_idx == 0:
                    current_floor_model, _, scene_cloud = stage_b_floor_estimation(frame, moge_model, device)

                # EDIT 2: SYNC FIX
                # We MUST append to this list EVERY frame, even if the model didn't update.
                # This ensures len(all_floor_models) == len(all_joints).
                all_floor_models.append(current_floor_model)

                # --- STAGE A: Pose Estimation (Every Frame) ---
                # [cite: 77] NLF must run every frame to capture the dance.
                joints3d, vertices3d = stage_a_nlf_implementation(frame, nlf_model, device=device)
                
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

                    # 5. Calculate Volume (Reuse verts_np)
                    try:
                        if verts_np.shape[0] > 3: # ConvexHull needs >3 points
                            hull = ConvexHull(verts_np)
                            current_vol = hull.volume
                        else:
                            current_vol = 0.0
                    except Exception:
                        current_vol = last_valid_volume
                    
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

    verify_pipeline_integrity(all_joints, all_volumes, current_floor_model)

# 1. Initialize Extractor with 55-frame window as per Turab et al. (2025)
    # [cite: 13, 149, 272]
    extractor = LMAExtractor(window_size=55, fps=fps)
    
    # 2. Extract 55 Feature Descriptors
    # [cite: 123] The pipeline extracts a descriptor vector composed of 55 features.
    lma_dict = extractor.extract_all_features(all_joints, all_volumes, all_floor_models)
    
    # 3. Flatten dictionary to a (Frames, 55) NumPy Array
    # This ensures your script moves from a dictionary to the list expected by ML models.
    feature_keys = sorted(lma_dict.keys())
    if len(feature_keys) != 55:
        print(f"[!] WARNING: Extracted {len(feature_keys)} features. Expected 55.")
    
    # Stack features into a matrix of shape (Total_Frames, 55)
    lma_matrix = np.stack([lma_dict[k] for k in feature_keys], axis=1)

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

    args = parser.parse_args()

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

    # 3. LOAD MODELS (Once per process)
    print("Loading Models...")
    nlf_model = torch.jit.load('models/nlf_l_multi_0.3.2.torchscript').to(device).eval()
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)

    # 4. RUN SEQUENTIALLY
    # (If run by external MP script, this list will just contain 1 item)
    for video_path in video_files:
        try:
            process_single_video(
                video_path, 
                args.output_dir, 
                nlf_model, 
                moge_model, 
                device, 
                viz=args.viz
            )
        except Exception as e:
            print(f"Failed on {video_path}: {e}")

if __name__ == "__main__":
    main()