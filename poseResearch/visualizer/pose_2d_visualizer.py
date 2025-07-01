import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from utils.visualizer import PoseVisualizer
from utils.skeleton_config import SkeletonConfig
from visualizer.skeleton_config import create_skeleton_config


class Pose2DVisualizer(PoseVisualizer):
    """Sophisticated 2D pose visualizer for flatpose outputs with configurable skeleton"""

    def __init__(
        self,
        visualize_every_n_batches: int = 1,
        save_plots: bool = True,
        output_dir: str = "./visualizations",
        show_labels: bool = False,
        max_people: int = 2,
        skeleton_config: SkeletonConfig = None,
        skeleton_type: str = "coco",
        create_videos: bool = False,
        video_fps: int = 15,
    ):
        # Initialize base class
        super().__init__(
            output_dir=output_dir, create_videos=create_videos, video_fps=video_fps
        )

        self.visualize_every_n_batches = visualize_every_n_batches
        self.save_plots = save_plots
        self.show_labels = show_labels
        self.max_people = max_people

        # Set up skeleton configuration (default to COCO for 2D)
        if skeleton_config is not None:
            self.skeleton_config = skeleton_config
        else:
            self.skeleton_config = create_skeleton_config(skeleton_type)

        # Cache skeleton properties for performance
        self._cache_skeleton_properties()

        # Fixed axis limits for consistent view
        self.fixed_axis_limits = None

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

    def compute_fixed_axis_limits(self, poses_2d: torch.Tensor, margin: int = 100):
        """
        Compute fixed axis limits from all poses to ensure consistent frame size

        Args:
            poses_2d: Tensor of shape (batch_size, num_frames, num_keypoints, 2)
            margin: Pixel margin around the poses
        """
        poses_np = poses_2d.detach().cpu().numpy()

        # Flatten to get all keypoints across all people and frames
        all_keypoints = poses_np.reshape(-1, 2)  # (total_keypoints, 2)

        # Remove invalid keypoints (NaN values)
        valid_keypoints = all_keypoints[~np.isnan(all_keypoints).any(axis=1)]

        if len(valid_keypoints) > 0:
            x_min, x_max = (
                valid_keypoints[:, 0].min() - margin,
                valid_keypoints[:, 0].max() + margin,
            )
            y_min, y_max = (
                valid_keypoints[:, 1].min() - margin,
                valid_keypoints[:, 1].max() + margin,
            )

            self.fixed_axis_limits = (x_min, x_max, y_min, y_max)
            print(
                f"Fixed axis limits: x=[{x_min:.0f}, {x_max:.0f}], y=[{y_min:.0f}, {y_max:.0f}]"
            )
        else:
            self.fixed_axis_limits = None
            print("No valid keypoints found for axis limits")

    def should_visualize(self, stage_name: str, batch_idx: int) -> bool:
        """Only visualize flatpose stages, every N batches"""
        return (
            stage_name == "flatpose" and batch_idx % self.visualize_every_n_batches == 0
        )

    def plot_single_pose_2d(self, ax, keypoints, person_id=0, title="", image=None):
        """
        Plot a single 2D pose with sophisticated styling

        Args:
            ax: Matplotlib axis
            keypoints: Array of shape (num_keypoints, 2) containing 2D keypoint coordinates
            person_id: ID of the person (for color variation)
            title: Title for the subplot
            image: Optional background image to overlay pose on
        """
        if len(keypoints) == 0:
            ax.text(
                0.5,
                0.5,
                "No pose detected",
                fontsize=12,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
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
                padding = np.full((self.num_keypoints - keypoints.shape[0], 2), np.nan)
                keypoints = np.vstack([keypoints, padding])
            else:
                keypoints = keypoints[: self.num_keypoints]

        # Show background image if provided
        if image is not None:
            ax.imshow(image, alpha=0.7)

        # Plot skeleton connections first (so they appear behind keypoints)
        for i, (start_idx, end_idx) in enumerate(self.skeleton_links):
            if (
                start_idx < len(keypoints)
                and end_idx < len(keypoints)
                and not np.isnan(keypoints[start_idx]).any()
                and not np.isnan(keypoints[end_idx]).any()
            ):

                x_coords = [keypoints[start_idx, 0], keypoints[end_idx, 0]]
                y_coords = [keypoints[start_idx, 1], keypoints[end_idx, 1]]
                color = self.skeleton_colors[i]

                ax.plot(x_coords, y_coords, c=color, linewidth=3, alpha=0.8, zorder=1)

        # Plot keypoints on top
        for i, (x, y) in enumerate(keypoints):
            if not np.isnan(x) and not np.isnan(y):  # Skip invalid keypoints
                color = self.keypoint_colors[i]

                # Draw keypoint with border for better visibility
                ax.scatter(
                    x,
                    y,
                    c=color,
                    s=80,
                    alpha=0.9,
                    edgecolors="white",
                    linewidth=2,
                    zorder=2,
                )

                # Optionally add keypoint labels
                if self.show_labels and i < len(self.keypoint_names):
                    ax.annotate(
                        f"{i}",
                        (x, y),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="black", alpha=0.7
                        ),
                    )

        # Set axis properties
        ax.set_xlabel("X coordinate (pixels)", fontsize=10)
        ax.set_ylabel("Y coordinate (pixels)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Invert Y axis for image coordinates (if showing image overlay)
        if image is not None:
            ax.invert_yaxis()

        # Set aspect ratio and grid
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Use fixed axis limits if available, otherwise calculate per frame
        if self.fixed_axis_limits is not None:
            x_min, x_max, y_min, y_max = self.fixed_axis_limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        elif len(keypoints) > 0:
            # Fallback to per-frame calculation if no fixed limits
            valid_keypoints = keypoints[~np.isnan(keypoints).any(axis=1)]
            if len(valid_keypoints) > 0:
                margin = 50  # pixels
                x_min, x_max = (
                    valid_keypoints[:, 0].min() - margin,
                    valid_keypoints[:, 0].max() + margin,
                )
                y_min, y_max = (
                    valid_keypoints[:, 1].min() - margin,
                    valid_keypoints[:, 1].max() + margin,
                )
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

    def visualize_2d_poses(
        self, poses_2d: torch.Tensor, batch_info: dict, stage_name: str
    ):
        """Visualize 2D poses from flatpose stage with sophisticated styling"""
        batch_idx = batch_info["batch_idx"]

        # Convert to numpy for visualization
        poses_np = poses_2d.detach().cpu().numpy()
        batch_size, num_frames, num_keypoints, _ = poses_np.shape

        # Validate input dimensions
        if num_keypoints != self.num_keypoints:
            print(
                f"Warning: Expected {self.num_keypoints} keypoints for {self.skeleton_config.__class__.__name__}, got {num_keypoints}"
            )

        # Create figure with subplots for multiple people
        fig = plt.figure(figsize=(8 * min(self.max_people, 2), 8))
        num_people = min(batch_size, self.max_people)

        if num_people == 0:
            plt.text(
                0.5,
                0.5,
                f"No poses detected in batch {batch_idx}",
                ha="center",
                va="center",
                transform=fig.transFigure,
                fontsize=16,
            )
        else:
            subplot_cols = min(num_people, 2)  # Max 2 columns
            subplot_rows = (num_people + 1) // 2 if num_people > 2 else 1

            for person_idx in range(min(num_people, 4)):  # Limit to 4 people max
                ax = fig.add_subplot(subplot_rows, subplot_cols, person_idx + 1)

                if person_idx < num_people and num_frames > 0:
                    keypoints = poses_np[
                        person_idx, 0
                    ]  # First frame, shape: (num_keypoints, 2)
                    title = (
                        f"Person {person_idx + 1} - {stage_name} - Batch {batch_idx}"
                    )
                    self.plot_single_pose_2d(
                        ax, keypoints, person_id=person_idx, title=title
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No person detected",
                        fontsize=12,
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(
                        f"Person {person_idx + 1} - {stage_name} - Batch {batch_idx}",
                        fontsize=12,
                        fontweight="bold",
                    )

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
            plt.savefig(
                f"{self.output_dir}/{stage_name}_batch_{batch_idx}_2d_{skeleton_name}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def visualize_3d_poses(
        self, poses_3d: torch.Tensor, batch_info: dict, stage_name: str
    ):
        """3D visualization not supported by 2D visualizer - skip"""
        pass

    def create_pose_overlay(
        self,
        poses_2d: torch.Tensor,
        background_image: np.ndarray,
        person_idx: int = 0,
        frame_idx: int = 0,
    ):
        """
        Create a pose overlay on a background image

        Args:
            poses_2d: Tensor of shape (batch_size, frames, num_keypoints, 2)
            background_image: Background image as numpy array
            person_idx: Which person to visualize
            frame_idx: Which frame to visualize

        Returns:
            Image with pose overlay
        """
        poses_np = poses_2d.detach().cpu().numpy()

        if person_idx >= poses_np.shape[0] or frame_idx >= poses_np.shape[1]:
            return background_image

        keypoints = poses_np[person_idx, frame_idx]  # Shape: (num_keypoints, 2)

        # Validate and pad/truncate keypoints if needed
        if keypoints.shape[0] != self.num_keypoints:
            if keypoints.shape[0] < self.num_keypoints:
                padding = np.full((self.num_keypoints - keypoints.shape[0], 2), np.nan)
                keypoints = np.vstack([keypoints, padding])
            else:
                keypoints = keypoints[: self.num_keypoints]

        image_overlay = background_image.copy()

        # Create color map for OpenCV (BGR format)
        color_map = {}
        for part, hex_color in self.body_part_colors.items():
            # Convert hex to BGR for OpenCV
            hex_color = hex_color.lstrip("#")
            rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            bgr = (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR
            color_map[part] = bgr

        # Draw skeleton connections
        for i, (start_idx, end_idx) in enumerate(self.skeleton_links):
            if (
                start_idx < len(keypoints)
                and end_idx < len(keypoints)
                and not np.isnan(keypoints[start_idx]).any()
                and not np.isnan(keypoints[end_idx]).any()
            ):

                start_point = tuple(map(int, keypoints[start_idx]))
                end_point = tuple(map(int, keypoints[end_idx]))

                # Get color for this skeleton link
                part = self.keypoint_body_parts[start_idx]
                color_bgr = color_map.get(part, (255, 255, 255))

                cv2.line(image_overlay, start_point, end_point, color_bgr, 3)

        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            if not np.isnan(x) and not np.isnan(y):
                point = (int(x), int(y))

                # Get color for this keypoint
                part = self.keypoint_body_parts[i]
                color_bgr = color_map.get(part, (255, 255, 255))

                cv2.circle(image_overlay, point, 6, color_bgr, -1)
                cv2.circle(image_overlay, point, 6, (255, 255, 255), 2)  # White border

        return image_overlay

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
        """Create videos from 2D pose visualizations"""
        if not self.create_videos:
            return []

        created_videos = []
        skeleton_name = self.skeleton_config.__class__.__name__.replace(
            "SkeletonConfig", ""
        ).lower()

        # Create video from 2D pose frames
        video_filename = f"2d_pose_animation_{skeleton_name}.mp4"
        video_path = self.create_video_from_images(video_filename, "*_2d_*.png")

        if video_path:
            created_videos.append(video_path)
            print(f"✅ Created 2D pose video: {video_path}")
        else:
            print("❌ Failed to create 2D pose video")

        return created_videos
