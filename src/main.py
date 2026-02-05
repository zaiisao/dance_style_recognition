import torch
import torchvision
import numpy as np
import cv2
from sklearn.linear_model import QuantileRegressor
from scipy.spatial import ConvexHull
from moge.model.v2 import MoGeModel
from tqdm import tqdm
import trimesh

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
        
def main():
    video_path = "/home/sogang/mnt/db_1/jaehoon/nsfw_datasets/sfw/kinetics-dataset/k700-2020/train/cartwheeling/_Ar-OVgeCvA_000004_000014.mp4" # Replaces single image path
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
    max_frames = int(fps * DEBUG_DURATION)
    print(f"--- DEBUG MODE: Processing only {max_frames} frames ---")

    all_joints = []
    all_volumes = []
    all_vertices = []
    all_floor_models = []
    scene_cloud = None

    last_valid_volume = 0.0

    with tqdm(total=min(total_frames, max_frames), desc="Processing Frames", unit="frame") as pbar:
        while cap.isOpened():
            if len(all_joints) >= max_frames:
                print(f"Reached limit of {max_frames} frames.")
                break

            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Floor Estimation
            floor_model, _, current_scene_cloud = stage_b_floor_estimation(frame, moge_model, device=device)
            all_floor_models.append(floor_model)

            scene_cloud = current_scene_cloud

            # 2. Pose Estimation
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
                # FIX: Array is (24, 3), so [0] gives the full Pelvis vector (x,y,z)
                pelvis_pos = joints_np[0] 
                
                # Predict Floor Y (Height) at Pelvis Z (Depth)
                floor_y = floor_model.predict(pelvis_pos[2].reshape(-1, 1))[0]
                height_above_floor = floor_y - pelvis_pos[1] # Y is Down
                
                pbar.set_postfix(h=f"{height_above_floor:.2f}m", vol=f"{current_vol:.3f}")
            else:
                pbar.set_postfix(status="No Det")
            
            pbar.update(1)
            
    cap.release()
    print("Video processing complete.")

    verify_pipeline_integrity(all_joints, all_volumes, floor_model)

    print("\n--- GENERATING VISUAL DEBUG ASSETS ---")

    # 2. Digital Twin Video
    # Note: Ensure 'all_joints' contains clean (N, 3) arrays, not None
    render_comprehensive_dashboard(
        video_path, 
        all_joints, 
        all_vertices, 
        all_floor_models, 
        scene_cloud,
        output_path="debug_short_clip.mp4"
    )

if __name__ == "__main__":
    main()