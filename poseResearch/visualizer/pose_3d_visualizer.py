from typing import Optional
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.visualizer import PoseVisualizer
from utils.skeleton_config import SkeletonConfig
from visualizer.skeleton_config import create_skeleton_config


class Pose3DVisualizer(PoseVisualizer):
    """Dedicated 3D pose visualizer for poselifting outputs with configurable skeleton"""

    def __init__(
        self,
        visualize_every_n_batches: int = 1,
        save_plots: bool = True,
        output_dir: str = "./visualizations",
        show_labels: bool = False,
        max_people: int = 2,
        skeleton_config: Optional[SkeletonConfig] = None,
        skeleton_type: str = "anatomical",
        create_videos: bool = False,
        video_fps: int = 15,
        fixed_axis: bool = True,
        save_path: str = "",
    ):
        # Initialize base class
        super().__init__(
            output_dir=output_dir, create_videos=create_videos, video_fps=video_fps
        )

        self.visualize_every_n_batches = visualize_every_n_batches
        self.save_plots = save_plots
        self.show_labels = show_labels
        self.max_people = max_people
        self.fixed_axis = fixed_axis

        # Store fixed axis limits (will be computed from first frame or all data)
        self.fixed_axis_limits = None

        # Set up skeleton configuration
        if skeleton_config is not None:
            self.skeleton_config = skeleton_config
        else:
            self.skeleton_config = create_skeleton_config(skeleton_type)

        # Cache skeleton properties for performance
        self._cache_skeleton_properties()
        self.save_path = save_path

    def _cache_skeleton_properties(self):
        """Cache skeleton properties for better performance during visualization"""
        self.keypoint_names = self.skeleton_config.get_keypoint_names()
        self.keypoint_id2name = self.skeleton_config.get_keypoint_id2name()
        self.keypoint_name2id = self.skeleton_config.get_keypoint_name2id()
        self.skeleton_links = self.skeleton_config.get_skeleton_links()
        self.body_part_colors = self.skeleton_config.get_body_part_colors()
        self.keypoint_body_parts = self.skeleton_config.get_keypoint_body_parts()
        self.keypoint_colors = self.skeleton_config.get_keypoint_colors()
        self.skeleton_colors = self.skeleton_config.get_skeleton_colors()
        self.num_keypoints = self.skeleton_config.get_num_keypoints()

    def should_visualize(self, stage_name: str, batch_idx: int) -> bool:
        """Only visualize poselifting stages, every N batches"""
        return (
            stage_name == "poselifting"
            and batch_idx % self.visualize_every_n_batches == 0
        )

    def compute_fixed_axis_limits(self, all_poses: torch.Tensor):
        """
        Set fixed axis limits from -3 to 3 for all axes

        Args:
            all_poses: Tensor of shape (people, frames, keypoints, 3)
        """
        if not self.fixed_axis:
            return

        # Set fixed axis limits from -3 to 3 for all axes
        self.fixed_axis_limits = {
            "x": (-3, 3),
            "y": (-3, 3),
            "z": (-3, 3),
        }

        print(
            f"Fixed axis limits set: X={self.fixed_axis_limits['x']}, Y={self.fixed_axis_limits['y']}, Z={self.fixed_axis_limits['z']}"
        )

    def plot_single_pose_3d(self, ax, keypoints, person_id=0, title=""):
        """
        Plot a single 3D pose with sophisticated styling

        Args:
            ax: Matplotlib 3D axis
            keypoints: Array of shape (num_keypoints, 3) containing 3D keypoint coordinates
            person_id: ID of the person (for color variation)
            title: Title for the subplot
        """
        if len(keypoints) == 0:
            ax.text(0, 0, 0, "No pose detected", fontsize=12, ha="center")
            ax.set_title(title)
            return

        keypoints = np.array(keypoints)

        # Validate keypoint dimensions
        if keypoints.shape[0] != self.num_keypoints:
            print(
                f"Warning: Expected {self.num_keypoints} keypoints, got {keypoints.shape[0]}"
            )
            # Pad or truncate as needed
            if keypoints.shape[0] < self.num_keypoints:
                padding = np.full((self.num_keypoints - keypoints.shape[0], 3), np.nan)
                keypoints = np.vstack([keypoints, padding])
            else:
                keypoints = keypoints[: self.num_keypoints]

        # Plot keypoints
        for i, (x, y, z) in enumerate(keypoints):
            if (
                not np.isnan(x) and not np.isnan(y) and not np.isnan(z)
            ):  # Skip invalid keypoints
                color = self.keypoint_colors[i]
                ax.scatter(
                    x, y, z, c=color, s=80, alpha=0.9, edgecolors="black", linewidth=1.5
                )

                # Optionally add keypoint labels
                if self.show_labels and i < len(self.keypoint_names):
                    ax.text(
                        x,
                        y,
                        z,
                        f"{i}:{self.keypoint_names[i]}",
                        fontsize=8,
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            alpha=0.8,
                            edgecolor="gray",
                        ),
                    )

        # Plot skeleton connections
        for i, (start_idx, end_idx) in enumerate(self.skeleton_links):
            if (
                start_idx < len(keypoints)
                and end_idx < len(keypoints)
                and not np.isnan(keypoints[start_idx]).any()
                and not np.isnan(keypoints[end_idx]).any()
            ):

                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                color = self.skeleton_colors[i]

                ax.plot(
                    [start_point[0], end_point[0]],
                    [start_point[1], end_point[1]],
                    [start_point[2], end_point[2]],
                    c=color,
                    linewidth=4,
                    alpha=0.8,
                )

        # Set axis properties
        ax.set_xlabel("X (mm)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y (mm)", fontsize=12, fontweight="bold")
        ax.set_zlabel("Z (mm)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Set axis limits - use fixed limits if available, otherwise compute from current pose
        if self.fixed_axis and self.fixed_axis_limits is not None:
            # Use pre-computed fixed axis limits
            ax.set_xlim(self.fixed_axis_limits["x"])
            ax.set_ylim(self.fixed_axis_limits["y"])
            ax.set_zlim(self.fixed_axis_limits["z"])
        else:
            # Fallback to fixed limits from -3 to 3 for all axes
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_zlim(-3, 3)

        # Set viewing angle for better visualization of human pose
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3)

        # Make the plot look more professional
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("gray")
        ax.yaxis.pane.set_edgecolor("gray")
        ax.zaxis.pane.set_edgecolor("gray")
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)

    def visualize_2d_poses(
        self, poses_2d: torch.Tensor, frame_info: dict, stage_name: str
    ):
        """2D visualization not supported by 3D visualizer"""
        pass  # Do nothing - this visualizer is only for 3D

    def visualize_3d_poses(
        self, poses_3d: torch.Tensor, frame_info: dict, stage_name: str
    ):
        """Visualize 3D poses from poselifting stage with professional styling"""
        frame_idx = frame_info["batch_idx"]

        # Convert to numpy for visualization
        poses_np = poses_3d.detach().cpu().numpy()
        batch_size, num_frames, num_keypoints, _ = poses_np.shape

        # Validate input dimensions
        if num_keypoints != self.num_keypoints:
            print(
                f"Warning: Expected {self.num_keypoints} keypoints for {self.skeleton_config.__class__.__name__}, got {num_keypoints}"
            )

        # Create figure with subplots for multiple people
        fig = plt.figure(figsize=(8 * min(self.max_people, 2), 9))
        num_people = min(batch_size, self.max_people)

        if num_people == 0:
            plt.text(
                0.5,
                0.5,
                f"No poses detected in frame {frame_idx}",
                ha="center",
                va="center",
                transform=fig.transFigure,
                fontsize=16,
            )
        else:
            subplot_cols = min(num_people, 2)  # Max 2 columns
            subplot_rows = (num_people + 1) // 2 if num_people > 2 else 1

            for person_idx in range(min(num_people, 4)):  # Limit to 4 people max
                ax = fig.add_subplot(
                    subplot_rows, subplot_cols, person_idx + 1, projection="3d"
                )

                if person_idx < num_people and num_frames > 0:
                    keypoints = poses_np[
                        person_idx, 0
                    ]  # First frame, shape: (num_keypoints, 3)
                    title = (
                        f"Person {person_idx + 1} - {stage_name} - Frame {frame_idx}"
                    )
                    self.plot_single_pose_3d(
                        ax, keypoints, person_id=person_idx, title=title
                    )
                else:
                    ax.text(0, 0, 0, "No person detected", fontsize=12, ha="center")
                    ax.set_title(
                        f"Person {person_idx + 1} - {stage_name} - Frame {frame_idx}",
                        fontsize=14,
                        fontweight="bold",
                    )
                    ax.set_xlabel("X (mm)", fontsize=12)
                    ax.set_ylabel("Y (mm)", fontsize=12)
                    ax.set_zlabel("Z (mm)", fontsize=12)

        # Add a legend for body parts
        legend_elements = []
        for part, color in self.body_part_colors.items():
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=part.replace("_", " ").title(),
                )
            )

        if num_people > 0:
            fig.legend(
                handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98)
            )

        plt.tight_layout()

        if self.save_plots:
            import os

            os.makedirs(self.output_dir, exist_ok=True)
            skeleton_name = self.skeleton_config.__class__.__name__.replace(
                "SkeletonConfig", ""
            ).lower()

            # Use batch_idx for sequential video frame naming
            filename = f"{self.output_dir}/{stage_name}_batch_{frame_idx:04d}_3d_{skeleton_name}.png"

            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def print_skeleton_info(self):
        """Print information about the current skeleton configuration"""
        self.skeleton_config.print_info()

    def get_skeleton_config(self) -> SkeletonConfig:
        """Get the current skeleton configuration"""
        return self.skeleton_config

    def set_skeleton_config(self, skeleton_config: SkeletonConfig):
        """Set a new skeleton configuration"""
        self.skeleton_config = skeleton_config
        self._cache_skeleton_properties()

    def set_skeleton_type(self, skeleton_type: str):
        """Set skeleton configuration by type name"""
        self.skeleton_config = create_skeleton_config(skeleton_type)
        self._cache_skeleton_properties()

    def create_all_videos(self) -> list:
        """Create videos from 3D pose visualizations"""
        if not self.create_videos:
            return []

        created_videos = []
        skeleton_name = self.skeleton_config.__class__.__name__.replace(
            "SkeletonConfig", ""
        ).lower()

        # Create video from 3D pose frames
        video_filename = f"pose3D_{self.save_path}_{skeleton_name}.mp4"
        video_path = self.create_video_from_images(video_filename, "*_3d_*.png")

        if video_path:
            created_videos.append(video_path)
            print(f"✅ Created 3D pose video: {video_path}")

        return created_videos
