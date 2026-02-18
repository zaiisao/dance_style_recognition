import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

def render_comprehensive_dashboard(video_path, all_joints, all_vertices, all_floor_models, scene_cloud, lma_features=None, output_path="dashboard_output.mp4"):
    # [cite: 75] 1920x1080 resolution is standard for the AIST++ dataset used in the paper.
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height * 2))
    fig = plt.figure(figsize=(20, 12), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1.2])

    # [cite: 123] The paper uses a vector of 55 features. We'll map the 4 global effort sums for the chart.
    lma_limit_buffer = 0.15
    lma_limits = {}
    if lma_features:
        for k in ['Effort_Weight_Global', 'Effort_Time_Global', 'Effort_Flow_Global', 'Effort_Space_Global']:
            vals = lma_features.get(k, [0])
            lma_limits[k] = (np.min(vals), np.max(vals) + lma_limit_buffer)

    frame_idx = 0
    total_frames = len(all_joints)
    
    with tqdm(total=total_frames, desc="Rendering Turab-Style Dashboard") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames: break
            
            fig.clf()
            
            # --- PANEL 1: FLOOR AWARE 3D SCENE [cite: 76, 83] ---
            ax_scene = fig.add_subplot(gs[0, 1], projection='3d')
            ax_scene.set_title("Floor Aware 3D Reconstruction", fontsize=14, fontweight='bold')
            
            # Draw Floor Grid (Lime wireframe as per methodology stage 2) [cite: 52]
            gx, gz = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(0, 5, 15))
            f_model = all_floor_models[frame_idx]
            gy = f_model.predict(gz.flatten().reshape(-1, 1)).reshape(gx.shape)
            ax_scene.plot_wireframe(gx, gz, -gy, color='lime', alpha=0.4, linewidth=0.5)

            # Draw Skeleton [cite: 77]
            joints = all_joints[frame_idx]
            if len(joints) > 0:
                ax_scene.scatter(joints[:,0], joints[:,2], -joints[:,1], c='red', s=15)
                # Skeleton Lines
                from visualizer import SKELETON_PARENTS
                for p1, p2 in SKELETON_PARENTS:
                    ax_scene.plot([joints[p1,0], joints[p2,0]], [joints[p1,2], joints[p2,2]], 
                                 [-joints[p1,1], -joints[p2,1]], c='black', alpha=0.6)

            ax_scene.view_init(elev=20, azim=45)
            ax_scene.set_axis_off()

            # --- PANEL 2: LMA EXPRESSIVE DYNAMICS [cite: 172, 272] ---
            # Recreating Fig 6 from paper: Long-term kinematic evolution
            ax_lma = fig.add_subplot(gs[1, 1])
            ax_lma.set_title("LMA Effort Evolution (Temporal Context)", fontsize=14, fontweight='bold')
            
            # Show sliding window of 120 frames to see patterns [cite: 168]
            win = 120
            s_idx, e_idx = max(0, frame_idx - win), frame_idx + 1
            time_axis = np.arange(s_idx, e_idx)
            
            efforts = [
                ('Effort_Weight_Global', 'Weight (Energy)', '#d62728'),
                ('Effort_Time_Global', 'Time (Suddenness)', '#1f77b4'),
                ('Effort_Flow_Global', 'Flow (Control)', '#2ca02c')
            ]
            
            for key, label, color in efforts:
                data = lma_features.get(key, np.zeros(total_frames))[s_idx:e_idx]
                ax_lma.plot(time_axis, data, label=label, color=color, linewidth=2.5)
            
            ax_lma.axvline(x=frame_idx, color='black', linestyle='--', alpha=0.5)
            ax_lma.legend(loc='upper left', fontsize=10)
            ax_lma.set_xlabel("Frame Number [cite: 273]", fontsize=12)
            ax_lma.set_ylabel("Normalized Magnitude", fontsize=12)
            ax_lma.grid(True, alpha=0.2)

            # Final Render and Overlay
            fig.canvas.draw()
            plot_img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            viz_layer = cv2.resize(plot_img, (width * 2, height * 2))
            
            # Stitch: [Video | 3D Scene] / [Body Model | LMA Charts]
            viz_layer[0:height, 0:width] = frame # Raw Video top-left
            writer.write(viz_layer)
            
            frame_idx += 1
            pbar.update(1)

    cap.release(); writer.release(); plt.close()