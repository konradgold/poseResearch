#!/usr/bin/env python3
"""
Lean example: Pipeline auto-detects where to start based on dataloader content.
"""

import torch
from utils.data_loader import DataLoader
from pipeline import EstimationPipe
from estimation.preprocess.preprocess_estimation import PreprocessEstimation
from estimation.pose2D.pose_estimation_2D import TwoDPoseEstimation
from estimation.pose3D.pose_estimation_3D import ThreeDPoseEstimation


# Minimal dummy estimators
class DummyPreprocessor(PreprocessEstimation):
    def _forward(self, data):
        return torch.randn(data.size(0), 224, 224, 3)

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "preprocessor"


class Dummy2DPose(TwoDPoseEstimation):
    def _forward(self, images):
        return torch.randn(2, images.size(0), 17, 3)

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "flatpose"


class Dummy3DPose(ThreeDPoseEstimation):
    def _forward(self, poses_2d):
        return poses_2d + torch.randn_like(poses_2d) * 0.1

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "poselifting"


def example_1_full_pipeline():
    """Full pipeline from scratch."""
    print("=== Full Pipeline ===")

    data_loader = DataLoader()
    pipeline = EstimationPipe(
        DummyPreprocessor(), Dummy2DPose(), Dummy3DPose(), data_loader
    )

    batches = [torch.randn(5, 480, 640, 3)]

    for result in pipeline.forward(batches):
        print(f"Final result: {result.shape}")

    data_loader.save_json("results.json")


def example_2_from_2d_poses():
    """Load 2D poses and auto-continue from 3D lifting."""
    print("\n=== Auto-start from 2D Poses ===")

    data_loader = DataLoader()
    data_loader.load_json("results.json")  # Has 2D poses

    # Same pipeline - auto-detects to start from poselifting
    pipeline = EstimationPipe(
        DummyPreprocessor(), Dummy2DPose(), Dummy3DPose(), data_loader
    )

    dummy_batch = [torch.randn(1, 1, 1, 1)]  # Won't be used

    for result in pipeline.forward(dummy_batch):
        print(f"3D from stored 2D: {result.shape}")


def example_3_manual_2d_input():
    """Manually add 2D poses and auto-continue."""
    print("\n=== Manual 2D Input ===")

    data_loader = DataLoader()

    # Manually add 2D poses
    poses_2d = torch.randn(3, 10, 17, 3)
    data_loader.handle(poses_2d, {"stage_name": "flatpose"})

    # Pipeline auto-detects and starts from poselifting
    pipeline = EstimationPipe(
        DummyPreprocessor(), Dummy2DPose(), Dummy3DPose(), data_loader
    )

    dummy_batch = [torch.randn(1, 1, 1, 1)]

    for result in pipeline.forward(dummy_batch):
        print(f"3D from manual 2D: {result.shape}")


if __name__ == "__main__":
    example_1_full_pipeline()
    example_2_from_2d_poses()
    example_3_manual_2d_input()
