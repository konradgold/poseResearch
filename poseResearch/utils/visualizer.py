from abc import ABC, abstractmethod
import torch
import os
import cv2
import glob
from typing import Optional


class PoseVisualizer(ABC):
    """Abstract base class for visualizing pose estimation outputs"""
    
    def __init__(self, output_dir: str = "./visualizations", create_videos: bool = False, video_fps: int = 15):
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
    def visualize_2d_poses(self, poses_2d: torch.Tensor, batch_info: dict, stage_name: str):
        """
        Visualize 2D pose estimations (flatpose output)
        
        Args:
            poses_2d: Tensor of shape (batch_size, frames, 17, 2) for 2D keypoints
            batch_info: Dictionary containing batch metadata
            stage_name: Name of the pipeline stage
        """
        pass
    
    @abstractmethod
    def visualize_3d_poses(self, poses_3d: torch.Tensor, batch_info: dict, stage_name: str):
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
    
    def create_video_from_images(self, video_filename: str, image_pattern: str = "*.png") -> Optional[str]:
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
        
        if not image_files:
            print(f"No images found matching pattern: {image_search_pattern}")
            return None
        
        print(f"Creating video from {len(image_files)} images...")
        
        # Read first image to get dimensions
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            print(f"Could not read first image: {image_files[0]}")
            return None
        
        height, width, layers = first_image.shape
        output_video_path = os.path.join(self.output_dir, video_filename)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, self.video_fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"Could not open video writer for {output_video_path}")
            return None
        
        # Add each image to video
        for i, image_file in enumerate(image_files):
            image = cv2.imread(image_file)
            if image is not None:
                video_writer.write(image)
                if i % 20 == 0:  # Progress update every 20 frames
                    print(f"Processing frame {i+1}/{len(image_files)}")
            else:
                print(f"Warning: Could not read image {image_file}")
        
        video_writer.release()
        print(f"Video saved to: {output_video_path}")
        
        # Track created videos
        self._created_videos.append(output_video_path)
        return output_video_path
    
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
    
    def get_created_videos(self) -> list:
        """Get list of all videos created by this visualizer"""
        return self._created_videos.copy()
    
    def cleanup_images_after_video(self, keep_sample: bool = True):
        """
        Clean up image files after video creation to save space
        
        Args:
            keep_sample: Whether to keep a few sample images
        """
        if not self.create_videos:
            return
        
        image_files = sorted(glob.glob(os.path.join(self.output_dir, "*.png")))
        
        if keep_sample:
            # Keep every 50th image as samples
            files_to_keep = image_files[::50][:5]  # Keep up to 5 sample images
            files_to_remove = [f for f in image_files if f not in files_to_keep]
        else:
            files_to_remove = image_files
        
        removed_count = 0
        for image_file in files_to_remove:
            try:
                os.remove(image_file)
                removed_count += 1
            except Exception as e:
                print(f"Could not remove {image_file}: {e}")
        
        if removed_count > 0:
            print(f"Cleaned up {removed_count} image files, kept {len(image_files) - removed_count} samples") 