import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ..utils.visualizer import PoseVisualizer


class Pose2DVisualizer(PoseVisualizer):
    """Sophisticated 2D pose visualizer for flatpose outputs"""
    
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
        """Setup skeleton structure and colors for 2D visualization"""
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
        
        # Define body part colors for more intuitive visualization
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
        """Only visualize flatpose stages, every N batches"""
        return (stage_name == "flatpose" and 
                batch_idx % self.visualize_every_n_batches == 0)
    
    def plot_single_pose_2d(self, ax, keypoints, person_id=0, title="", image=None):
        """
        Plot a single 2D pose with sophisticated styling
        
        Args:
            ax: Matplotlib axis
            keypoints: Array of shape (17, 2) containing 2D keypoint coordinates
            person_id: ID of the person (for color variation)
            title: Title for the subplot
            image: Optional background image to overlay pose on
        """
        if len(keypoints) == 0:
            ax.text(0.5, 0.5, "No pose detected", fontsize=12, ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(title)
            return
            
        keypoints = np.array(keypoints)
        
        # Show background image if provided
        if image is not None:
            ax.imshow(image, alpha=0.7)
        
        # Plot skeleton connections first (so they appear behind keypoints)
        for i, (start_idx, end_idx) in enumerate(self.skeleton_links):
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                not np.isnan(keypoints[start_idx]).any() and not np.isnan(keypoints[end_idx]).any()):
                
                x_coords = [keypoints[start_idx, 0], keypoints[end_idx, 0]]
                y_coords = [keypoints[start_idx, 1], keypoints[end_idx, 1]]
                color = self.skeleton_colors[i]
                
                ax.plot(x_coords, y_coords, c=color, linewidth=3, alpha=0.8, zorder=1)
        
        # Plot keypoints on top
        for i, (x, y) in enumerate(keypoints):
            if not np.isnan(x) and not np.isnan(y):  # Skip invalid keypoints
                color = self.keypoint_colors[i]
                
                # Draw keypoint with border for better visibility
                ax.scatter(x, y, c=color, s=80, alpha=0.9, edgecolors='white', 
                          linewidth=2, zorder=2)
                
                # Optionally add keypoint labels
                if self.show_labels and i < len(self.keypoint_names):
                    ax.annotate(f'{i}', (x, y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8, 
                              color='white', fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # Set axis properties
        ax.set_xlabel('X coordinate (pixels)', fontsize=10)
        ax.set_ylabel('Y coordinate (pixels)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Invert Y axis for image coordinates (if showing image overlay)
        if image is not None:
            ax.invert_yaxis()
        
        # Set aspect ratio and grid
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits based on keypoints
        if len(keypoints) > 0:
            valid_keypoints = keypoints[~np.isnan(keypoints).any(axis=1)]
            if len(valid_keypoints) > 0:
                margin = 50  # pixels
                x_min, x_max = valid_keypoints[:, 0].min() - margin, valid_keypoints[:, 0].max() + margin
                y_min, y_max = valid_keypoints[:, 1].min() - margin, valid_keypoints[:, 1].max() + margin
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

    def visualize_2d_poses(self, poses_2d: torch.Tensor, batch_info: dict, stage_name: str):
        """Visualize 2D poses from flatpose stage with sophisticated styling"""
        batch_idx = batch_info["batch_idx"]
        
        # Convert to numpy for visualization
        poses_np = poses_2d.detach().cpu().numpy()
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
                ax = fig.add_subplot(subplot_rows, subplot_cols, person_idx + 1)
                
                if person_idx < num_people and num_frames > 0:
                    keypoints = poses_np[person_idx, 0]  # First frame, shape: (17, 2)
                    title = f"Person {person_idx + 1} - {stage_name} - Batch {batch_idx}"
                    self.plot_single_pose_2d(ax, keypoints, person_id=person_idx, title=title)
                else:
                    ax.text(0.5, 0.5, "No person detected", fontsize=12, ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(f"Person {person_idx + 1} - {stage_name} - Batch {batch_idx}",
                               fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if self.save_plots:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(f'{self.output_dir}/{stage_name}_batch_{batch_idx}_2d_detailed.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_3d_poses(self, poses_3d: torch.Tensor, batch_info: dict, stage_name: str):
        """3D visualization not supported by 2D visualizer - skip"""
        pass
    
    def create_pose_overlay(self, poses_2d: torch.Tensor, background_image: np.ndarray, 
                           person_idx: int = 0, frame_idx: int = 0):
        """
        Create a pose overlay on a background image
        
        Args:
            poses_2d: Tensor of shape (batch_size, frames, 17, 2)
            background_image: Background image as numpy array
            person_idx: Which person to visualize
            frame_idx: Which frame to visualize
            
        Returns:
            Image with pose overlay
        """
        poses_np = poses_2d.detach().cpu().numpy()
        
        if person_idx >= poses_np.shape[0] or frame_idx >= poses_np.shape[1]:
            return background_image
        
        keypoints = poses_np[person_idx, frame_idx]  # Shape: (17, 2)
        image_overlay = background_image.copy()
        
        # Draw skeleton connections
        for i, (start_idx, end_idx) in enumerate(self.skeleton_links):
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                not np.isnan(keypoints[start_idx]).any() and not np.isnan(keypoints[end_idx]).any()):
                
                start_point = tuple(map(int, keypoints[start_idx]))
                end_point = tuple(map(int, keypoints[end_idx]))
                
                # Convert color name to BGR for OpenCV
                color_name = self.skeleton_colors[i]
                color_map = {
                    'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0),
                    'orange': (0, 165, 255), 'purple': (128, 0, 128), 'brown': (42, 42, 165)
                }
                color_bgr = color_map.get(color_name, (255, 255, 255))
                
                cv2.line(image_overlay, start_point, end_point, color_bgr, 3)
        
        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            if not np.isnan(x) and not np.isnan(y):
                point = (int(x), int(y))
                
                # Convert color name to BGR
                color_name = self.keypoint_colors[i]
                color_bgr = color_map.get(color_name, (255, 255, 255))
                
                cv2.circle(image_overlay, point, 6, color_bgr, -1)
                cv2.circle(image_overlay, point, 6, (255, 255, 255), 2)  # White border
        
        return image_overlay 