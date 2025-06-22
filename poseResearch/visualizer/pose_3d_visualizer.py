import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from ..utils.visualizer import PoseVisualizer


class Pose3DVisualizer(PoseVisualizer):
    """Dedicated 3D pose visualizer for poselifting outputs"""
    
    def __init__(self, visualize_every_n_batches: int = 1, save_plots: bool = True, 
                 output_dir: str = "./visualizations", show_labels: bool = False,
                 max_people: int = 2):
        self.visualize_every_n_batches = visualize_every_n_batches
        self.save_plots = save_plots
        self.output_dir = output_dir
        self.show_labels = show_labels
        self.max_people = max_people
        self.setup_skeleton()
        
    def setup_skeleton(self):
        """Setup skeleton structure and colors for 3D visualization"""
        # COCO 17 keypoint connections for skeleton visualization
        self.skeleton_links = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
            (5, 11), (6, 12), (11, 12),  # torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # legs
        ]
        
        # Keypoint names (COCO 17)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Define body part colors for more intuitive 3D visualization
        self.body_part_colors = {
            'head': 'red',
            'torso': 'blue', 
            'left_arm': 'green',
            'right_arm': 'orange',
            'left_leg': 'purple',
            'right_leg': 'brown'
        }
        
        # Map keypoints to body parts
        self.keypoint_body_parts = [
            'head', 'head', 'head', 'head', 'head',  # 0-4: face
            'torso', 'torso', 'left_arm', 'right_arm',  # 5-8: shoulders, elbows
            'left_arm', 'right_arm', 'torso', 'torso',  # 9-12: wrists, hips
            'left_leg', 'right_leg', 'left_leg', 'right_leg'  # 13-16: knees, ankles
        ]
        
        # Set up colors for keypoints based on body parts
        self.keypoint_colors = [self.body_part_colors[part] for part in self.keypoint_body_parts]
        
        # Set up colors for skeleton links
        self.skeleton_colors = []
        for start_idx, end_idx in self.skeleton_links:
            # Use the color of the first keypoint in the connection
            color = self.keypoint_colors[start_idx]
            self.skeleton_colors.append(color)
    
    def should_visualize(self, stage_name: str, batch_idx: int) -> bool:
        """Only visualize poselifting stages, every N batches"""
        return (stage_name == "poselifting" and 
                batch_idx % self.visualize_every_n_batches == 0)
    
    def plot_single_pose_3d(self, ax, keypoints, person_id=0, title=""):
        """
        Plot a single 3D pose with sophisticated styling
        
        Args:
            ax: Matplotlib 3D axis
            keypoints: Array of shape (17, 3) containing 3D keypoint coordinates
            person_id: ID of the person (for color variation)
            title: Title for the subplot
        """
        if len(keypoints) == 0:
            ax.text(0, 0, 0, "No pose detected", fontsize=12, ha='center')
            ax.set_title(title)
            return
            
        keypoints = np.array(keypoints)
        
        # Plot keypoints
        for i, (x, y, z) in enumerate(keypoints):
            if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):  # Skip invalid keypoints
                color = self.keypoint_colors[i]
                ax.scatter(x, y, z, c=color, s=60, alpha=0.9, edgecolors='black', linewidth=1)
                
                # Optionally add keypoint labels
                if self.show_labels and i < len(self.keypoint_names):
                    ax.text(x, y, z, f'{i}:{self.keypoint_names[i][:3]}', fontsize=7, 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Plot skeleton connections
        for i, (start_idx, end_idx) in enumerate(self.skeleton_links):
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                not np.isnan(keypoints[start_idx]).any() and not np.isnan(keypoints[end_idx]).any()):
                
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                color = self.skeleton_colors[i]
                
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       [start_point[2], end_point[2]], 
                       c=color, linewidth=3, alpha=0.8)
        
        # Set axis properties
        ax.set_xlabel('X (mm)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (mm)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Z (mm)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        # Set equal aspect ratio and proper limits
        if len(keypoints) > 0:
            valid_keypoints = keypoints[~np.isnan(keypoints).any(axis=1)]
            if len(valid_keypoints) > 0:
                max_range = np.array([
                    valid_keypoints[:,0].max() - valid_keypoints[:,0].min(), 
                    valid_keypoints[:,1].max() - valid_keypoints[:,1].min(),
                    valid_keypoints[:,2].max() - valid_keypoints[:,2].min()
                ]).max() / 2.0
                
                mid_x = (valid_keypoints[:,0].max() + valid_keypoints[:,0].min()) * 0.5
                mid_y = (valid_keypoints[:,1].max() + valid_keypoints[:,1].min()) * 0.5
                mid_z = (valid_keypoints[:,2].max() + valid_keypoints[:,2].min()) * 0.5
                
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)
        
        # Make the plot look more professional
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)

    def visualize_2d_poses(self, poses_2d: torch.Tensor, batch_info: dict, stage_name: str):
        """2D visualization not supported by 3D visualizer"""
        pass  # Do nothing - this visualizer is only for 3D
    
    def visualize_3d_poses(self, poses_3d: torch.Tensor, batch_info: dict, stage_name: str):
        """Visualize 3D poses from poselifting stage with professional styling"""
        batch_idx = batch_info["batch_idx"]
        
        # Convert to numpy for visualization
        poses_np = poses_3d.detach().cpu().numpy()
        batch_size, num_frames, num_keypoints, _ = poses_np.shape
        
        # Create figure with subplots for multiple people
        fig = plt.figure(figsize=(8 * min(self.max_people, 2), 8))
        num_people = min(batch_size, self.max_people)
        
        if num_people == 0:
            plt.text(0.5, 0.5, f"No poses detected in batch {batch_idx}", 
                    ha='center', va='center', transform=fig.transFigure, fontsize=16)
        else:
            subplot_cols = min(num_people, 2)  # Max 2 columns
            subplot_rows = (num_people + 1) // 2 if num_people > 2 else 1
            
            for person_idx in range(min(num_people, 4)):  # Limit to 4 people max
                ax = fig.add_subplot(subplot_rows, subplot_cols, person_idx + 1, projection='3d')
                
                if person_idx < num_people and num_frames > 0:
                    keypoints = poses_np[person_idx, 0]  # First frame, shape: (17, 3)
                    title = f"Person {person_idx + 1} - {stage_name} - Batch {batch_idx}"
                    self.plot_single_pose_3d(ax, keypoints, person_id=person_idx, title=title)
                else:
                    ax.text(0, 0, 0, "No person detected", fontsize=12, ha='center')
                    ax.set_title(f"Person {person_idx + 1} - {stage_name} - Batch {batch_idx}",
                               fontsize=13, fontweight='bold')
                    ax.set_xlabel('X (mm)', fontsize=11)
                    ax.set_ylabel('Y (mm)', fontsize=11)
                    ax.set_zlabel('Z (mm)', fontsize=11)
        
        plt.tight_layout()
        
        if self.save_plots:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(f'{self.output_dir}/{stage_name}_batch_{batch_idx}_3d_dedicated.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show() 