#!/usr/bin/env python3
"""
Simple example usage of pose visualization with video creation
"""

import torch
import numpy as np
import json
from visualizer.pose_3d_visualizer import Pose3DVisualizer


def main():
    try:
        with open("results_interactive4_t3-cam16_anatomical.json", "r") as f:
            data = json.load(f)

        poses_list = data["poses_3d"]
        metadata = data["metadata"]

        print(
            f"✅ Loaded {metadata['num_people']} people, {metadata['num_frames']} frames"
        )

        # Convert to tensor
        poses = torch.from_numpy(np.array(poses_list)).float()

        # Create visualizer with video enabled
        visualizer = Pose3DVisualizer(
            skeleton_type="anatomical",
            output_dir="./pose_video_output",
            create_videos=True,
            video_fps=30,
        )

        # Generate all frames and create video in one call
        created_videos = visualizer.visualize_all_frames(
            poses, "h36m_poses", "pose_data"
        )

        if created_videos:
            print(f"Success! Video created: {created_videos[0]}")
        else:
            print("Failed to create video")

    except FileNotFoundError:
        print("❌ Pose data file not found")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
