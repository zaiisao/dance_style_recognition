import torch
import torchvision
import numpy as np
import cv2
from sklearn.linear_model import QuantileRegressor
from scipy.spatial import ConvexHull
from moge.model.v2 import MoGeModel
from tqdm import tqdm
import trimesh

from lma_extractor import LMAExtractor
from visualizer import render_comprehensive_dashboard

def stage_a_nlf_implementation(frame, model, device="cuda"):
    # Preprocess image: Convert OpenCV (H, W, C) -> Torch (C, H, W)
    # Replaces: image = torchvision.io.read_image(image_path).to(device)
    if isinstance(frame, np.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).to(device)
    else:
        frame_tensor = frame.to(device)

    # Maintain original batching logic
    frame_batch = frame_tensor.unsqueeze(0).float() / 255.0

    with torch.inference_mode():
        # Using the batched detection method highlighted in your script 
        pred = model.detect_smpl_batched(frame_batch)
    
    # [cite_start]Extract absolute 3D data required for LMA components [cite: 1, 10]
    # 'joints3d' for Body/Effort/Space; 'vertices3d' for Shape (Volume)
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
        
import numpy as np
import matplotlib.pyplot as plt

def verify_lma_integrity(npy_path):
    print(f"--- VERIFYING: {npy_path} ---")
    
    # [FIX] Load Dictionary format properly
    try:
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception:
        print("[!] Fatal: Could not load dictionary. File might be corrupted.")
        return

    # Basic setup
    keys = list(data.keys())
    n_frames = len(data[keys[0]])
    print(f"Loaded {n_frames} frames. Keys: {keys}")
    
    # ---------------------------------------------------------
    # 1. PHYSICAL UNIT CHECK (Checking 'weight' / Kinetic Energy)
    # ---------------------------------------------------------
    # Weight = Speed^2. 
    # If units are mm, Speed is 1000x -> Weight is 1,000,000x larger.
    
    weight_feat = data.get('weight', np.zeros(n_frames))
    max_val = np.max(weight_feat)
    
    print("\n[1] Physical Unit Scaling Check")
    if max_val > 1000:
        print(f"    [!] WARNING: Max Energy is {max_val:.2f}.")
        print("        Likely Unit Error: Data seems to be in MILLIMETERS.")
        print("        LMA standard is METERS. Divide inputs by 1000.0 before extraction.")
    elif max_val < 1e-4:
        print(f"    [!] WARNING: Max value is {max_val:.2e}. Data is vanishingly small.")
    else:
        print(f"    [OK] Value range looks consistent with Meters/Seconds.")
        print(f"         Max Energy Value: {max_val:.3f}")

    # ---------------------------------------------------------
    # 2. GEOMETRIC LOGIC CHECK (Checking 'space')
    # ---------------------------------------------------------
    # Space = Path_Length / Displacement.
    # By definition, this MUST be >= 1.0 (Triangle Inequality).
    
    space_feat = data.get('space', np.zeros(n_frames))
    
    print("\n[2] Geometric Logic Check")
    if np.any(space_feat < 0.99): # Allow tiny float error
        print(f"    [!] FAIL: Found Space ratios < 1.0. (Min: {np.min(space_feat):.3f})")
    elif np.isnan(space_feat).any():
        print("    [!] FAIL: NaNs found. (Division by zero in Displacement?)")
    else:
        print("    [OK] Geometry looks valid (All Space Ratios >= 1.0).")

    # ---------------------------------------------------------
    # 3. SHAPE FLOW CHECK
    # ---------------------------------------------------------
    # Note: Our extractor saved 'Shape Flow' (Derivative), not raw Volume.
    # So we check if the derivative is non-zero (proving the volume changed).
    
    shape_flow = data.get('shape', np.zeros(n_frames))
    
    print("\n[3] Body Volume Flow Check")
    if np.allclose(shape_flow, 0):
        print("    [!] FAIL: Shape Flow is flat zero. Convex Hull might have failed.")
    else:
        print(f"    [OK] Volume changes detected (Min flow: {np.min(shape_flow):.4f}, Max flow: {np.max(shape_flow):.4f})")

    # ---------------------------------------------------------
    # 4. TEMPORAL DYNAMICS CHECK
    # ---------------------------------------------------------
    print("\n[4] Temporal Dynamics Check")
    dead_count = 0
    for k in keys:
        var = np.var(data[k])
        if var < 1e-6:
            print(f"    [!] WARNING: Feature '{k}' is static (Zero Variance).")
            dead_count += 1
            
    if dead_count == 0:
        print("    [OK] All features show temporal evolution.")

    # ---------------------------------------------------------
    # 5. VISUALIZATION
    # ---------------------------------------------------------
    print("\n[5] Generating Visualization...")
    plt.figure(figsize=(12, 6))
    
    # Plot normalized features to compare trends
    for k in keys:
        sig = data[k]
        if np.std(sig) > 1e-6:
            # Z-score normalization for cleaner plotting
            norm_sig = (sig - np.mean(sig)) / np.std(sig)
            plt.plot(norm_sig, label=k, alpha=0.8, linewidth=1.5)
    
    plt.title("LMA Feature Evolution (Normalized)")
    plt.xlabel("Frame")
    plt.ylabel("Z-Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = "lma_verification_plot.png"
    plt.savefig(output_img)
    print(f"    Saved plot to {output_img}")

def main():
    video_path = "/home/sogang/mnt/db_1/jaehoon/aist/gBR_sBM_c01_d04_mBR0_ch01.mp4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Models...")
    nlf_model = torch.jit.load('models/nlf_l_multi_0.3.2.torchscript').to(device).eval()
    # [cite_start]Load MoGe-v2 (The authors chose MoGe for ground surface quality [cite: 85])
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    DEBUG_DURATION = 3.0 
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0

    all_joints = []
    all_volumes = []
    all_vertices = []
    all_floor_models = []
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
                    verts_np = vertices3d[0].cpu().numpy()
                    joints_np = joints3d[0].cpu().numpy()
                    
                    if verts_np.ndim == 3:
                        verts_np = verts_np[0]
                    if joints_np.ndim == 3:
                        joints_np = joints_np[0]
                    
                    if np.mean(np.abs(verts_np[:, 2])) > 10.0: 
                        verts_np /= 1000.0
                        joints_np /= 1000.0
                    
                    all_vertices.append(verts_np)

                    # Volume Calculation
                    try:
                        # Trimesh is robust, but don't force 'is_watertight' for open clothing meshes
                        mesh = trimesh.convex.convex_hull(verts_np)
                        current_vol = mesh.volume
                        last_valid_volume = current_vol
                    except Exception:
                        # If it fails, rely on the last known good volume (gap filling)
                        pass
                else:
                    all_vertices.append(None)
                
                if joints_np is not None:
                    all_joints.append(joints_np)
                else:
                    all_joints.append([]) # Keep list length consistent
                
                all_volumes.append(current_vol)

                # --- METRICS ---
                if joints_np is not None:
                    pelvis_pos = joints_np[0] 
                    
                    # Predict Floor Y (Height) at Pelvis Z (Depth)
                    floor_y = current_floor_model.predict(pelvis_pos[2].reshape(-1, 1))[0]
                    height_above_floor = floor_y - pelvis_pos[1]
                    
                    pbar.set_postfix(h=f"{height_above_floor:.2f}m", vol=f"{current_vol:.3f}")
                else:
                    pbar.set_postfix(status="No Det")
                
                frame_idx += 1
                pbar.update(1)
                
        cap.release()
        print("Video processing complete.")

    verify_pipeline_integrity(all_joints, all_volumes, current_floor_model)

    # 1. Initialize Extractor
    extractor = LMAExtractor(window_size=55, fps=fps)
    
    # 2. Extract Features
    # Pass 'all_floor_models' so the extractor can fix the camera shake/tilt.
    lma_features = extractor.extract_all_features(all_joints, all_volumes, all_floor_models)
    
    print(f"[-] Feature Extraction Complete")
    print(f"    Input Frames:  {len(all_joints)}")
    
    # [FIX] Handle Dictionary output (No .shape attribute)
    print(f"    Output Keys:   {list(lma_features.keys())}")
    
    # Check length of the first feature (e.g., 'weight') to confirm sequence length
    first_key = list(lma_features.keys())[0]
    print(f"    Seq Length:    {len(lma_features[first_key])}")

    # 3. Save for Classification
    output_filename = "lma_features_output.npy"
    
    # [NOTE] np.save wraps the dict in a generic object array. 
    # When loading later, use: data = np.load(..., allow_pickle=True).item()
    np.save(output_filename, lma_features)
    print(f"    Saved features to: {output_filename}")
    
    verify_lma_integrity("lma_features_output.npy")

    print("\n--- GENERATING VISUAL DEBUG ASSETS ---")
    render_comprehensive_dashboard(
        video_path, 
        all_joints, 
        all_vertices, 
        all_floor_models, 
        scene_cloud,
        lma_features=lma_features,
        output_path="debug_short_clip.mp4"
    )

if __name__ == "__main__":
    main()