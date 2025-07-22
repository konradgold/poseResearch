#!/usr/bin/env python3
"""
Example showing how to visualize poses using ProcessManager and the new visualization system
"""

import argparse
from pathlib import Path
from utils.process_manager import ProcessManager
from visualizer.pose_3d_visualizer import Pose3DVisualizer
from visualizer.pose_2d_visualizer import Pose2DVisualizer

pr_dir = Path(__file__).parent


def visualize_2d_poses(path: str):
    data_loader = ProcessManager()
    data_loader.load_json(
        pr_dir
        / "dataloader"
        / f"results{'_' if path != '' else ''}{path}_flatpose.json"
    )

    visualizer_2d = Pose2DVisualizer(
        save_path=path,
        skeleton_type="anatomical",
        output_dir="./pose_video_output_2d",
        create_videos=True,
        video_fps=30,
    )

    visualizer_2d.visualize_from_dataloader(data_loader, "flatpose")


def visualize_3d_poses(path: str):
    data_loader = ProcessManager()
    data_loader.load_json(
        pr_dir
        / "dataloader"
        / f"results{'_' if path != '' else ''}{path}_poselifting.json"
    )

    visualizer_3d = Pose3DVisualizer(
        save_path=path,
        skeleton_type="anatomical",
        output_dir="./pose_video_output_3d",
        create_videos=True,
        video_fps=30,
    )

    visualizer_3d.visualize_from_dataloader(data_loader, "poselifting")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 2D and 3D poses from processed data"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="none-yolo-motionbert-11m-20",
        help="Path identifier for the data files (default: none-yolo-motionbert-11m-20)",
    )
    parser.add_argument(
        "--only-2d", action="store_true", help="Only visualize 2D poses"
    )
    parser.add_argument(
        "--only-3d", action="store_true", help="Only visualize 3D poses"
    )

    args = parser.parse_args()

    if args.only_2d:
        print(f"Visualizing 2D poses for path: {args.path}")
        visualize_2d_poses(args.path)
    elif args.only_3d:
        print(f"Visualizing 3D poses for path: {args.path}")
        visualize_3d_poses(args.path)
    else:
        print(f"Visualizing both 2D and 3D poses for path: {args.path}")
        visualize_2d_poses(args.path)
        visualize_3d_poses(args.path)


if __name__ == "__main__":
    main()
