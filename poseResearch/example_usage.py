#!/usr/bin/env python3
"""
Lean example: DataLoader manages all input data, pipeline just runs stages.
"""

import torch
from utils.data_loader import DataLoader
from pipeline import EstimationPipe
from estimation.preprocess.preprocess_estimation import PreprocessEstimation
from estimation.preprocess.no_preprocess import NoPreprocess
from estimation.pose2D.pose_estimation_2D import TwoDPoseEstimation
from estimation.pose2D.yolo_estimation import YOLOEstimation
from estimation.pose3D.pose_estimation_3D import ThreeDPoseEstimation


# Minimal dummy estimators
class DummyPreprocessor(PreprocessEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "preprocessor"

    def _forward(self, data):
        return torch.randn(data.size(0), 224, 224, 3)


class Dummy2DPose(TwoDPoseEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "flatpose"

    def _forward(self, images):
        return torch.randn(2, images.size(0), 17, 3)


class Dummy3DPose(ThreeDPoseEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "poselifting"

    def _forward(self, poses_2d):
        return poses_2d


def example_1_full_pipeline():
    """Full pipeline from raw frames."""
    print("=== Full Pipeline ===")

    data_loader = DataLoader(save_path="results.json")
    pipeline = EstimationPipe(
        DummyPreprocessor(), Dummy2DPose(), Dummy3DPose(), data_loader
    )

    # Set input data in dataloader
    raw_frames = torch.randn(5, 480, 640, 3)
    data_loader.set_input(raw_frames)

    # Run pipeline - no parameters needed!
    result = pipeline.forward()
    print(f"Final result: {result.shape}")


def example_2_from_2d_poses():
    """Load 2D poses and auto-continue from 3D lifting."""
    print("\n=== Auto-start from 2D Poses ===")

    data_loader = DataLoader()
    data_loader.load_json("results.json")  # Has 2D poses

    # Pipeline auto-detects and starts from poselifting
    pipeline = EstimationPipe(
        DummyPreprocessor(), Dummy2DPose(), Dummy3DPose(), data_loader
    )

    # No input needed - uses stored 2D poses
    result = pipeline.forward()
    print(f"3D from stored 2D: {result.shape}")


def example_3_from_3d_poses():
    """Load 3D poses and auto-continue"""
    print("\n=== 3D Poses Input ===")

    data_loader = DataLoader()
    data_loader.load_json("dataloader/results_3d.json")

    # Pipeline auto-detects and starts from poselifting
    pipeline = EstimationPipe(
        DummyPreprocessor(), Dummy2DPose(), Dummy3DPose(), data_loader
    )

    result = pipeline.forward()
    print(f"3D from manual 2D: {result.shape}")


def example_4_individual_stage():
    """Use dataloader's run_stage method directly."""
    print("\n=== Individual Stage Usage ===")

    data_loader = DataLoader()

    # Add 2D poses
    poses_2d = torch.randn(2, 8, 17, 3)
    data_loader.handle(poses_2d, {"stage_name": "flatpose"})

    # Run only 3D lifting stage
    pose_3d_model = Dummy3DPose()
    result = data_loader.run_stage(pose_3d_model, "flatpose")
    print(f"Direct stage result: {result.shape}")


def example_5_from_video():
    """Load video and run pipeline."""
    print("\n=== Video Input ===")

    data_loader = DataLoader(save_path="results-from-video.json")
    data_loader.set_input_from_video("fem1_t1_preview.mp4", num_frames=20)

    print("Data loading complete.")

    pipeline = EstimationPipe(
        NoPreprocess(), YOLOEstimation("yolo11s-pose.pt"), Dummy3DPose(), data_loader
    )

    result = pipeline.forward()
    print(f"Video result: {result.shape}")


if __name__ == "__main__":
    # example_1_full_pipeline()
    # example_2_from_2d_poses()
    # example_3_from_3d_poses()
    # example_4_individual_stage()
    example_5_from_video()
