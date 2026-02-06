import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from tqdm import tqdm

SKELETON_PARENTS = [
    (0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9),
    (9,12), (12,15), (9,13), (9,14), (13,16), (14,17), (16,18), 
    (17,19), (18,20), (19,21)
]

def render_comprehensive_dashboard(video_path, all_joints, all_vertices, all_floor_models, scene_cloud, lma_features=None, output_path="dashboard_output.mp4"):
    print(f"Generating Dashboard: {output_path}...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_size = (width * 2, height * 2)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)
    
    fig = plt.figure(figsize=(16, 10), dpi=80)
    gs = fig.add_gridspec(2, 2)

    # [NEW] Pre-calculate Global Limits for LMA charts
    # This ensures the Y-axis doesn't jitter while the video plays.
    lma_limits = {}
    if lma_features is not None:
        for key, val in lma_features.items():
            vmin, vmax = np.min(val), np.max(val)
            margin = (vmax - vmin) * 0.1 if vmax != vmin else 1.0
            lma_limits[key] = (vmin - margin, vmax + margin)
    
    # Pre-calculate Grid X/Z
    gx = np.linspace(-3, 3, 20)
    gz = np.linspace(0, 6, 20)
    GX, GZ = np.meshgrid(gx, gz)
    
    frame_idx = 0
    # Use the length of the VIDEO or Joints, whichever is safer
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=len(all_joints), desc="Rendering Dashboard", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # [FIX 1] Safe Indexing. If lists are out of sync, stop or pad.
            if frame_idx >= len(all_joints):
                break
                
            joints = all_joints[frame_idx]
            
            # Safe access for vertices
            verts = None
            if frame_idx < len(all_vertices):
                verts = all_vertices[frame_idx]
            
            # Safe access for floor
            # Defaults to flat floor if missing
            floor_data = None
            if frame_idx < len(all_floor_models):
                floor_data = all_floor_models[frame_idx]

            # [FIX 2] Hybrid Floor Math (Handles both Models and Raw Numbers)
            if floor_data is not None:
                if hasattr(floor_data, 'predict'):
                    # It is a Scikit-Learn Model (Old way)
                    GY = floor_data.predict(GZ.flatten().reshape(-1, 1)).reshape(GX.shape)
                    floor_mean = np.mean(GY)
                elif isinstance(floor_data, (list, np.ndarray, tuple)) and len(floor_data) >= 2:
                    # It is [Slope, Intercept] (Smoothed way)
                    slope, intercept = floor_data[:2]
                    GY = (slope * GZ) + intercept
                    floor_mean = np.mean(GY)
                else:
                    # Fallback
                    GY = np.zeros_like(GX) + 1.5
                    floor_mean = 1.5
            else:
                GY = np.zeros_like(GX) + 1.5
                floor_mean = 1.5

            fig.clf()
            
            # === PANEL 2: SCENE & FLOOR ===
            ax1 = fig.add_subplot(gs[0, 1], projection='3d')
            ax1.set_title("Scene Reconstruction")

            if scene_cloud is not None:
                step = max(1, len(scene_cloud) // 2000) # Auto-downsample
                sc = scene_cloud[::step]
                ax1.scatter(sc[:,0], sc[:,2], -sc[:,1], s=1, c='gray', alpha=0.3)

            ax1.plot_wireframe(GX, GZ, -GY, color='lime', alpha=0.6, linewidth=0.5)
            
            if joints is not None and len(joints) > 0:
                # Handle possible (1, N, 3) shape
                j_plot = np.array(joints)
                if j_plot.ndim == 3: j_plot = j_plot[0]
                ax1.scatter(j_plot[:,0], j_plot[:,2], -j_plot[:,1], c='red', s=10)

            ax1.view_init(elev=30, azim=45)
            ax1.set_xlim(-2, 2); ax1.set_ylim(0, 5); ax1.set_zlim(-floor_mean-1, -floor_mean+2)
            ax1.set_axis_off()

            # === PANEL 3: BODY MESH ===
            ax2 = fig.add_subplot(gs[1, 0], projection='3d')
            ax2.set_title("Body Model (SMPL)")
            
            if verts is not None and len(verts) > 0:
                v_plot = np.array(verts)
                if v_plot.ndim == 3: v_plot = v_plot[0]
                
                v_sparse = v_plot[::20] 
                ax2.scatter(v_sparse[:,0], v_sparse[:,2], -v_sparse[:,1], c='blue', s=2, alpha=0.5)
                ax2.plot_wireframe(GX, GZ, -GY, color='lime', alpha=0.2)
            
            ax2.view_init(elev=20, azim=135) 
            ax2.set_xlim(-1, 1); ax2.set_ylim(0, 5); ax2.set_zlim(-floor_mean-1, -floor_mean+2)
            ax2.set_axis_off()

            # === PANEL 4 (Bottom Right): LMA LIVE CHARTS ===
            if lma_features is not None:
                # Create a nested 4-row grid inside the bottom-right slot
                gs_lma = gs[1, 1].subgridspec(4, 1, hspace=0.1)
                
                plots = [
                    ('weight', 'Weight (Energy)', 'red'),
                    ('time', 'Time (Suddenness)', 'blue'),
                    ('flow', 'Flow (Control)', 'green'),
                    ('space', 'Space (Directness)', 'purple')
                ]
                
                # Sliding window logic (Show last 60 frames)
                window_size = 60 
                start_idx = max(0, frame_idx - window_size)
                end_idx = frame_idx + 1
                
                for i, (key, label, color) in enumerate(plots):
                    ax_lma = fig.add_subplot(gs_lma[i, 0])
                    data = lma_features.get(key, np.zeros(total_frames))
                    
                    # 1. Plot History Line
                    ax_lma.plot(np.arange(start_idx, end_idx), data[start_idx:end_idx], 
                                color=color, linewidth=2)
                    
                    # 2. Plot Current Frame Dot
                    if frame_idx < len(data):
                        ax_lma.scatter(frame_idx, data[frame_idx], color='black', s=20, zorder=5)

                    # 3. Styling
                    ax_lma.set_xlim(start_idx, start_idx + window_size)
                    if key in lma_limits:
                        ax_lma.set_ylim(lma_limits[key])
                        
                    ax_lma.set_ylabel(key.capitalize(), fontsize=8, rotation=0, labelpad=20)
                    ax_lma.set_facecolor('#f8f9fa')
                    ax_lma.grid(True, alpha=0.3)
                    
                    # Hide X-axis labels for the top 3 charts
                    if i < 3:
                        ax_lma.tick_params(axis='x', bottom=False, labelbottom=False)
                    else:
                        ax_lma.set_xlabel("Frame Window", fontsize=8)
            else:
                # Fallback if no features provided
                ax3 = fig.add_subplot(gs[1, 1])
                ax3.text(0.5, 0.5, "No LMA Data", ha='center')

            # Render
            fig.canvas.draw()
            plot_img = np.asarray(fig.canvas.buffer_rgba())
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
            viz_layer = cv2.resize(plot_img, (width * 2, height * 2))
            
            viz_layer[0:height, 0:width] = frame
            cv2.line(viz_layer, (width, 0), (width, height*2), (0,0,0), 2)
            cv2.line(viz_layer, (0, height), (width*2, height), (0,0,0), 2)

            writer.write(viz_layer)
            
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    plt.close()
    print(f"Dashboard saved to {output_path}")
