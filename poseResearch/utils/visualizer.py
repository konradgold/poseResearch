from abc import ABC, abstractmethod
import torch
import os
import cv2
import glob
from typing import Optional, Literal, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from .process_manager import ProcessManager

# Define visualization stage types
VisualizationStage = Literal["flatpose", "poselifting"]


class PoseVisualizer(ABC):
    """Abstract base class for visualizing pose estimation outputs"""

    def __init__(
        self,
        output_dir: str = "./visualizations",
        create_videos: bool = False,
        video_fps: int = 15,
    ):
        """
        Initialize base visualizer

        Args:
            output_dir: Directory to save visualizations
            create_videos: Whether to create videos from generated images
            video_fps: Frame rate for video creation
        """
        self.output_dir = output_dir
        self.create_videos = create_videos
        self.video_fps = video_fps
        self._created_videos = []

    @abstractmethod
    def visualize_2d_poses(
        self, poses_2d: torch.Tensor, batch_info: dict, stage_name: str
    ):
        """
        Visualize 2D pose estimations (flatpose output)

        Args:
            poses_2d: Tensor of shape (batch_size, frames, 17, 2) for 2D keypoints
            batch_info: Dictionary containing batch metadata
            stage_name: Name of the pipeline stage
        """
        pass

    @abstractmethod
    def visualize_3d_poses(
        self, poses_3d: torch.Tensor, batch_info: dict, stage_name: str
    ):
        """
        Visualize 3D pose estimations (poselifting output)

        Args:
            poses_3d: Tensor of shape (batch_size, frames, 17, 3) for 3D keypoints
            batch_info: Dictionary containing batch metadata
            stage_name: Name of the pipeline stage
        """
        pass

    @abstractmethod
    def should_visualize(self, stage_name: str, batch_idx: int) -> bool:
        """
        Determine if visualization should be performed for this stage/batch

        Args:
            stage_name: Name of the pipeline stage
            batch_idx: Index of the current batch

        Returns:
            bool: Whether to perform visualization
        """
        pass

    def create_video_from_images(
        self, video_filename: str, image_pattern: str = "*.png"
    ) -> Optional[str]:
        """
        Create MP4 video from images in the output directory

        Args:
            video_filename: Name of the output video file
            image_pattern: Glob pattern to match image files (default: "*.png")

        Returns:
            Path to created video file, or None if creation failed
        """
        if not self.create_videos:
            return None

        if not os.path.exists(self.output_dir):
            print(f"Output directory does not exist: {self.output_dir}")
            return None

        # Find all matching images
        image_search_pattern = os.path.join(self.output_dir, image_pattern)
        image_files = sorted(glob.glob(image_search_pattern))

        # Additional safeguard: try to sort numerically by extracting frame numbers

        def extract_frame_number(filename):
            # Extract frame number from patterns like "frame_0001" or "batch_1"
            match = re.search(r"(?:frame_|batch_)(\d+)", os.path.basename(filename))
            return int(match.group(1)) if match else 0

        image_files = sorted(image_files, key=extract_frame_number)

        if not image_files:
            print(f"No images found matching pattern: {image_search_pattern}")
            return None

        print(f"Creating video from {len(image_files)} images...")
        print(f"First few images: {[os.path.basename(f) for f in image_files[:3]]}")

        # Read first image to get dimensions
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            print(f"Could not read first image: {image_files[0]}")
            return None

        height, width, layers = first_image.shape
        output_video_path = os.path.join(self.output_dir, video_filename)

        # Create video writer with a more compatible codec
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(
            output_video_path, fourcc, self.video_fps, (width, height)
        )

        if not video_writer.isOpened():
            print(f"Could not open video writer for {output_video_path}")
            return None

        # Add each image to video
        frames_written = 0
        for i, image_file in enumerate(image_files):
            image = cv2.imread(image_file)
            if image is not None:
                # Ensure image dimensions match the video writer
                if image.shape[:2] != (height, width):
                    print(
                        f"Resizing image {i} from {image.shape[:2]} to {(height, width)}"
                    )
                    image = cv2.resize(image, (width, height))

                video_writer.write(image)
                frames_written += 1
                if i % 5 == 0:  # More frequent progress updates
                    print(
                        f"Processing frame {i+1}/{len(image_files)} (written: {frames_written})"
                    )
            else:
                print(f"Warning: Could not read image {image_file}")

        print(f"Total frames written to video: {frames_written}")

        video_writer.release()

        # Verify the video file was created and has content
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            print(
                f"Video saved to: {output_video_path} (size: {os.path.getsize(output_video_path)} bytes)"
            )
            # Track created videos
            self._created_videos.append(output_video_path)
            return output_video_path
        else:
            print(f"Failed to create video file or file is empty: {output_video_path}")
            return None

    def create_all_videos(self) -> list:
        """
        Create videos for all visualization types (to be overridden by subclasses)

        Returns:
            List of created video file paths
        """
        if not self.create_videos:
            return []

        created_videos = []

        # Default: create a single video from all PNG files
        video_path = self.create_video_from_images("pose_animation.mp4")
        if video_path:
            created_videos.append(video_path)

        return created_videos

    def visualize_all_frames(
        self, poses: torch.Tensor, source_name: str, stage_name: str = "poses"
    ):
        """
        Visualize all frames in a pose tensor and optionally create video

        Args:
            poses: Tensor of shape (people, frames, keypoints, 2/3)
            source_name: Name for this visualization session
            stage_name: Stage name for file naming
        """
        num_people, num_frames = poses.shape[0], poses.shape[1]
        print(f"Processing {num_people} people across {num_frames} frames...")

        # Compute fixed axis limits if this visualizer supports it
        if hasattr(self, "compute_fixed_axis_limits"):
            self.compute_fixed_axis_limits(poses)

        # Process each frame individually
        for frame_idx in range(num_frames):
            # Extract single frame: (people, 1, keypoints, 2/3)
            frame_poses = poses[:, frame_idx : frame_idx + 1, :, :]

            # Create batch info for this frame
            batch_info = {
                "batch_idx": frame_idx,
                "source": source_name,
                "num_people": num_people,
                "num_frames": 1,
                "frame_id": frame_idx,
            }

            # Visualize based on pose dimensions
            if poses.shape[3] == 2:  # 2D poses
                self.visualize_2d_poses(frame_poses, batch_info, stage_name)
            elif poses.shape[3] == 3:  # 3D poses
                self.visualize_3d_poses(frame_poses, batch_info, stage_name)

            if frame_idx % 50 == 0:  # Progress update every 50 frames
                print(f"Processed frame {frame_idx+1}/{num_frames}")

        print(f"All frames visualized and saved to {self.output_dir}")

        # Create video if enabled
        if self.create_videos:
            created_videos = self.create_all_videos()
            if created_videos:
                print(f"Successfully created videos: {created_videos}")
                self.cleanup_images_after_video()
                return created_videos

        return []

    def get_created_videos(self) -> list:
        """Get list of all videos created by this visualizer"""
        return self._created_videos.copy()

    def cleanup_images_after_video(self):
        """
        Clean up image files after video creation to save space

        Args:
            keep_sample: Whether to keep a few sample images
        """
        if not self.create_videos:
            return

        image_files = sorted(glob.glob(os.path.join(self.output_dir, "*.png")))

        files_to_remove = image_files

        removed_count = 0
        for image_file in files_to_remove:
            try:
                os.remove(image_file)
                removed_count += 1
            except Exception as e:
                print(f"Could not remove {image_file}: {e}")

        if removed_count > 0:
            print(
                f"Cleaned up {removed_count} image files, kept {len(image_files) - removed_count} samples"
            )

    def visualize_from_dataloader(
        self,
        data_loader: "ProcessManager",
        stage: VisualizationStage,
        source_name: str = "dataloader_viz",
    ) -> Optional[list]:
        """
        Visualize poses from a ProcessManager for a specific stage

        Args:
            data_loader: ProcessManager containing the pose data
            stage: Which stage to visualize ("flatpose" or "poselifting")
            source_name: Name for this visualization session

        Returns:
            List of created video paths if videos were created, None otherwise
        """
        # Get tensor data from the specified stage
        poses_tensor = data_loader.get_tensor(stage)

        if poses_tensor is None:
            raise ValueError(f"No data found for stage '{stage}' in ProcessManager")

        print(f"Visualizing {stage} data with shape: {poses_tensor.shape}")

        # Validate tensor dimensions based on stage
        if stage == "flatpose":
            if poses_tensor.dim() != 4 or poses_tensor.shape[3] != 3:
                raise ValueError(
                    f"Expected flatpose data to have shape (people, frames, keypoints, 3), got {poses_tensor.shape}"
                )
            # For 2D visualization, we only use the first 2 coordinates
            poses_for_viz = poses_tensor[..., :2]  # (people, frames, keypoints, 2)

        elif stage == "poselifting":
            if poses_tensor.dim() != 4 or poses_tensor.shape[3] != 3:
                raise ValueError(
                    f"Expected poselifting data to have shape (people, frames, keypoints, 3), got {poses_tensor.shape}"
                )
            poses_for_viz = poses_tensor  # (people, frames, keypoints, 3)

        # Use the existing visualize_all_frames method
        created_videos = self.visualize_all_frames(poses_for_viz, source_name, stage)

        return created_videos
