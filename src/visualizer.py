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

def render_comprehensive_dashboard(video_path, all_joints, all_vertices, all_floor_models, scene_cloud, output_path="dashboard_output.mp4"):
    print(f"Generating Dashboard: {output_path}...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_size = (width * 2, height * 2)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)
    
    fig = plt.figure(figsize=(12, 12), dpi=80)
    
    # Pre-calculate Grid X/Z (The mesh structure doesn't change, only Y values)
    gx = np.linspace(-3, 3, 20)
    gz = np.linspace(0, 6, 20)
    GX, GZ = np.meshgrid(gx, gz)
    
    frame_idx = 0
    with tqdm(total=len(all_joints), desc="Rendering Dashboard", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= len(all_joints):
                break
                
            joints = all_joints[frame_idx]
            verts = all_vertices[frame_idx]
            
            # Get the specific floor model for THIS frame
            current_floor_model = all_floor_models[frame_idx]
            
            # Recalculate Grid Y for this frame's camera angle
            GY = current_floor_model.predict(GZ.flatten().reshape(-1, 1)).reshape(GX.shape)
            floor_mean = np.mean(GY)

            fig.clf()
            
            # === PANEL 2: SCENE & FLOOR ===
            ax1 = fig.add_subplot(222, projection='3d')
            ax1.set_title("Scene Reconstruction")

            if scene_cloud is not None:
                step = 10 
                sc = scene_cloud[::step]
                ax1.scatter(sc[:,0], sc[:,2], -sc[:,1], s=1, c='gray', alpha=0.3)

            # Plot UPDATED Floor
            ax1.plot_wireframe(GX, GZ, -GY, color='lime', alpha=0.6, linewidth=0.5)
            
            if joints is not None and len(joints) > 0:
                ax1.scatter(joints[:,0], joints[:,2], -joints[:,1], c='red', s=10)

            ax1.view_init(elev=30, azim=45)
            ax1.set_xlim(-2, 2); ax1.set_ylim(0, 5); ax1.set_zlim(-floor_mean-1, -floor_mean+2)
            ax1.set_axis_off()

            # === PANEL 3: BODY MESH ===
            ax2 = fig.add_subplot(223, projection='3d')
            ax2.set_title("Body Model (SMPL)")
            
            if verts is not None:
                v_sparse = verts[::20] 
                ax2.scatter(v_sparse[:,0], v_sparse[:,2], -v_sparse[:,1], c='blue', s=2, alpha=0.5)
                # Plot UPDATED Floor
                ax2.plot_wireframe(GX, GZ, -GY, color='lime', alpha=0.2)
            
            ax2.view_init(elev=20, azim=135) 
            ax2.set_xlim(-1, 1); ax2.set_ylim(0, 5); ax2.set_zlim(-floor_mean-1, -floor_mean+2)
            ax2.set_axis_off()

            # === PANEL 4: PHYSICS CHECK ===
            ax3 = fig.add_subplot(224, projection='3d')
            ax3.set_title("Physics Check (Side View)")
            
            # Plot UPDATED Floor
            ax3.plot_wireframe(GX, GZ, -GY, color='lime', linewidth=1.0)
            
            if joints is not None and len(joints) > 0:
                xs, ys, zs = joints[:,0], joints[:,1], joints[:,2]
                ax3.scatter(xs, zs, -ys, c='red', s=30)
                for p1, p2 in SKELETON_PARENTS:
                     ax3.plot([xs[p1], xs[p2]], [zs[p1], zs[p2]], [-ys[p1], -ys[p2]], c='blue', linewidth=2)

            ax3.view_init(elev=0, azim=90) 
            ax3.set_xlim(-2, 2); ax3.set_ylim(0, 5); ax3.set_zlim(-floor_mean-0.5, -floor_mean+2.0)
            ax3.set_yticks([]) 

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