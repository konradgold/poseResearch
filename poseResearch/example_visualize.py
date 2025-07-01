#!/usr/bin/env python3
"""
Example showing how to visualize poses using DataLoader and the new visualization system
"""

import torch
import numpy as np
import json
from pathlib import Path
from utils.data_loader import DataLoader
from visualizer.pose_3d_visualizer import Pose3DVisualizer
from visualizer.pose_2d_visualizer import Pose2DVisualizer


def visualize_2d_poses():
    data_loader = DataLoader()
    data_loader.load_json("dataloader/results_flatpose.json")

    visualizer_2d = Pose2DVisualizer(
        skeleton_type="coco",
        output_dir="./pose_video_output_2d",
        create_videos=True,
        video_fps=30,
    )

    visualizer_2d.visualize_from_dataloader(data_loader, "flatpose")


def visualize_3d_poses():
    data_loader = DataLoader()
    data_loader.load_json("dataloader/results_3d.json")

    visualizer_3d = Pose3DVisualizer(
        skeleton_type="anatomical",
        output_dir="./pose_video_output_3d",
        create_videos=True,
        video_fps=30,
    )

    visualizer_3d.visualize_from_dataloader(data_loader, "poselifting")


if __name__ == "__main__":
    # visualize_2d_poses()
    visualize_3d_poses()
