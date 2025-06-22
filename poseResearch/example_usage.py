#!/usr/bin/env python3
"""
Example usage of the pose estimation pipeline with specialized visualizers
"""

import torch
import numpy as np
import json
import os
import cv2
import glob
from pipeline import EstimationPipe
from visualizer.pose_3d_visualizer import Pose3DVisualizer
from visualizer.skeleton_config import create_skeleton_config


# Mock estimation stages for demonstration
class MockPreprocessor:
    def __init__(self):
        self.config = {"stage": "preprocessing"}
        
    def forward(self, x):
        print("Preprocessing stage")
        return x

class MockFlatpose:
    def __init__(self):
        self.config = {"stage": "flatpose"}
        
    def forward(self, x):
        print("Flatpose (2D) estimation stage")
        batch_size, frames = x.shape[0], x.shape[1] if len(x.shape) > 1 else 1
        return torch.randn(batch_size, frames, 17, 2)  # Mock 2D poses

class MockPoselifting:
    def __init__(self):
        self.config = {"stage": "poselifting"}
        
    def forward(self, x):
        print("Poselifting (3D) estimation stage")
        batch_size, frames = x.shape[0], x.shape[1]
        return torch.randn(batch_size, frames, 17, 3)  # Mock 3D poses

class MockOutputSaver:
    def handle(self, data, config):
        # Mock output saver - just prints info
        print(f"Saving output from {config.get('stage', 'unknown')} stage: {data.shape}")


def read_poses_from_anatomical_json(filepath: str) -> torch.Tensor:
    """Read 3D pose data from anatomical JSON file (H36M converted format)"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        poses_list = data["poses_3d"]
        metadata = data["metadata"]
        
        print(f"Loaded poses from {filepath}")
        print(f"Dataset: {metadata.get('original_dataset', 'unknown')}")
        print(f"People: {metadata['num_people']}")
        print(f"Frames: {metadata['num_frames']}")
        print(f"Keypoints: {metadata['num_keypoints']}")
        print(f"Skeleton type: {metadata['skeleton_type']}")
        
        poses_np = np.array(poses_list)
        print(f"Pose array shape: {poses_np.shape}")
        
        return torch.from_numpy(poses_np).float()
    except Exception as e:
        print(f"Error reading anatomical JSON file {filepath}: {e}")
        return None


def create_video_from_images(image_dir: str, output_video_path: str, fps: int = 10):
    """Create MP4 video from images in a directory"""
    print(f"Creating video from images in {image_dir}...")
    
    # Find all PNG images in the directory
    image_pattern = os.path.join(image_dir, "*.png")
    image_files = sorted(glob.glob(image_pattern))
    
    if not image_files:
        print(f"No PNG images found in {image_dir}")
        return None
    
    print(f"Found {len(image_files)} images")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Could not read first image: {image_files[0]}")
        return None
    
    height, width, layers = first_image.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Could not open video writer for {output_video_path}")
        return None
    
    # Add each image to the video
    for i, image_file in enumerate(image_files):
        image = cv2.imread(image_file)
        if image is not None:
            video_writer.write(image)
            if i % 10 == 0:  # Progress update every 10 frames
                print(f"Processing frame {i+1}/{len(image_files)}")
        else:
            print(f"Warning: Could not read image {image_file}")
    
    # Release the video writer
    video_writer.release()
    
    print(f"Video saved to: {output_video_path}")
    return output_video_path


def visualize_poses_3d_with_video(poses_3d: torch.Tensor, source_name: str, skeleton_type: str = "anatomical", fps: int = 10):
    """Visualize 3D poses and create video from all frames"""
    print(f"Visualizing poses from {source_name}...")
    
    output_dir = f"./pose_visualizations/{source_name}"
    
    # Create 3D visualizer with video creation enabled
    visualizer_3d = Pose3DVisualizer(
        skeleton_type=skeleton_type,
        visualize_every_n_batches=1,
        save_plots=True,
        output_dir=output_dir,
        show_labels=True,
        max_people=min(poses_3d.shape[0], 4),  # Limit to 4 people for visualization
        create_videos=True,  # Enable video creation
        video_fps=fps
    )
    
    # Print skeleton info
    print(f"Using {skeleton_type} skeleton:")
    visualizer_3d.print_skeleton_info()
    
    num_people, num_frames = poses_3d.shape[0], poses_3d.shape[1]
    print(f"Processing {num_people} people across {num_frames} frames...")
    
    # Process each frame individually to create separate images
    for frame_idx in range(num_frames):
        # Extract single frame: (people, 1, keypoints, 3)
        frame_poses = poses_3d[:, frame_idx:frame_idx+1, :, :]
        
        # Create batch info for this frame
        batch_info = {
            "batch_idx": frame_idx,
            "source": source_name,
            "num_people": num_people,
            "num_frames": 1,
            "frame_id": frame_idx
        }
        
        # Visualize this frame
        visualizer_3d.visualize_3d_poses(frame_poses, batch_info, "anatomical_data")
        
        if frame_idx % 50 == 0:  # Progress update every 50 frames
            print(f"Processed frame {frame_idx+1}/{num_frames}")
    
    print(f"All frames visualized and saved to {output_dir}")
    
    # Create video using visualizer's built-in video creation
    created_videos = visualizer_3d.create_all_videos()
    created_video = created_videos[0] if created_videos else None
    
    if created_video:
        print(f"Successfully created pose animation video: {created_video}")
        
        # Optionally clean up images to save space (keep some samples)
        print("Cleaning up image files to save space...")
        visualizer_3d.cleanup_images_after_video(keep_sample=True)
    
    return output_dir, created_video


def visualize_poses_3d(poses_3d: torch.Tensor, source_name: str, skeleton_type: str = "anatomical"):
    """Visualize 3D poses using the 3D visualizer (single frame version)"""
    print(f"Visualizing poses from {source_name}...")
    
    # Create 3D visualizer
    visualizer_3d = Pose3DVisualizer(
        skeleton_type=skeleton_type,
        visualize_every_n_batches=1,
        save_plots=True,
        output_dir=f"./pose_visualizations/{source_name}",
        show_labels=True,
        max_people=min(poses_3d.shape[0], 4)  # Limit to 4 people for visualization
    )
    
    # Print skeleton info
    print(f"Using {skeleton_type} skeleton:")
    visualizer_3d.print_skeleton_info()
    
    # Create batch info for visualization
    batch_info = {
        "batch_idx": 0,
        "source": source_name,
        "num_people": poses_3d.shape[0],
        "num_frames": poses_3d.shape[1]
    }
    
    # Visualize the poses
    visualizer_3d.visualize_3d_poses(poses_3d, batch_info, "anatomical_data")
    
    print(f"Visualization saved to ./pose_visualizations/{source_name}/")


def example_visualize_h36m_anatomical():
    """Example reading and visualizing H36M anatomical JSON format with video creation"""
    print("=== H36M Anatomical Visualization with Video Example ===")
    
    found_file = "results_interactive4_t3-cam16_anatomical.json"
    
    if found_file:
        print(f"Found anatomical file: {found_file}")
        poses_3d = read_poses_from_anatomical_json(found_file)
        
        if poses_3d is not None:
            # Create video from all frames (this will take longer but creates animation)
            output_dir, video_path = visualize_poses_3d_with_video(
                poses_3d, 
                "h36m_anatomical", 
                skeleton_type="anatomical",
                fps=15  # 15 FPS for smooth animation
            )
            
            if video_path:
                print(f"✅ Successfully created pose animation: {video_path}")
            else:
                print("❌ Failed to create video")
        else:
            print("Failed to load pose data")

def example_basic_pipeline_with_video():
    """Basic pipeline usage example with video creation"""
    print("\n=== Basic Pipeline with Video Example ===")
    
    # Create 3D visualizer with video creation enabled
    visualizer_3d = Pose3DVisualizer(
        skeleton_type="anatomical",
        save_plots=True,
        output_dir="./pipeline_visualizations",
        visualize_every_n_batches=1,
        create_videos=True,  # Enable video creation
        video_fps=10
    )
    
    # Create pipeline (no video parameters needed in pipeline anymore)
    pipeline = EstimationPipe(
        preprocessor=MockPreprocessor(),
        flatpose=MockFlatpose(),
        poselifting=MockPoselifting(),
        output_saver=MockOutputSaver(),
        visualizer_3d=visualizer_3d
    )
    
    # Create mock dataloader (multiple batches to create video frames)
    mock_dataloader = [
        torch.randn(2, 5, 256, 256, 3) for _ in range(10)  # 10 batches, each with 5 frames
    ]
    
    # Run pipeline
    results = []
    for result in pipeline.forward(mock_dataloader):
        results.append(result)
        print(f"Processed batch with shape: {result.shape}")
    
    print(f"Pipeline processed {len(results)} batches")
    
    # Create videos from all generated visualizations
    created_videos = pipeline.create_videos_from_visualizations()
    
    return results, created_videos


def example_skeleton_comparison_with_video():
    """Example comparing different skeleton configurations with video creation"""
    print("\n=== Skeleton Configuration Comparison with Videos ===")
    
    # Create sample poses with more frames for better video
    poses_3d = torch.randn(2, 30, 17, 3) * 0.5  # 30 frames for smoother video
    
    # Visualize with different skeleton types
    skeleton_types = ["anatomical", "coco"]
    
    for skeleton_type in skeleton_types:
        print(f"\nVisualizing with {skeleton_type} skeleton:")
        config = create_skeleton_config(skeleton_type)
        config.print_info()
        
        # Create video for this skeleton type
        output_dir, video_path = visualize_poses_3d_with_video(
            poses_3d, 
            f"comparison_{skeleton_type}", 
            skeleton_type=skeleton_type,
            fps=12
        )
        
        if video_path:
            print(f"✅ Created {skeleton_type} skeleton video: {video_path}")
        else:
            print(f"❌ Failed to create {skeleton_type} skeleton video")


def cleanup_sample_files():
    """Clean up sample files created for demonstration"""
    sample_files = [
        "sample_anatomical_poses.json"
    ]
    
    for filepath in sample_files:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Cleaned up: {filepath}")
            except Exception as e:
                print(f"Could not remove {filepath}: {e}")


if __name__ == "__main__":
    print("H36M Anatomical Pose Visualization with Video Creation")
    print("=" * 60)
    
    # Run main example - H36M anatomical data with video
    example_visualize_h36m_anatomical()
    
    # Uncomment below to also run pipeline example with video creation
    # print("\n" + "=" * 60)
    # example_basic_pipeline_with_video()
    
    # Uncomment below to run skeleton comparison with videos
    # print("\n" + "=" * 60) 
    # example_skeleton_comparison_with_video()
    
    print("\nVisualization and video creation completed!") 