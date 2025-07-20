#!/usr/bin/env python3
"""
Lean example: ProcessManager manages all input data, pipeline just runs stages.
"""

import argparse
import sys
import torch
from utils.process_manager import ProcessManager
from pipeline import EstimationPipe
from estimation.preprocess.preprocess_estimation import PreprocessEstimation
from estimation.pose2D.pose_estimation_2D import TwoDPoseEstimation
from estimation.pose3D.pose_estimation_3D import ThreeDPoseEstimation


# Minimal dummy estimators
class DummyPreprocessor(PreprocessEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "DummyPreprocessor"

    def _forward(self, data):  # type: ignore
        return torch.randn(data.size(0), 224, 224, 3)


class Dummy2DPose(TwoDPoseEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "Dummy2DPose"

    def _forward(self, images):
        return torch.randn(2, images.size(0), 17, 3)


class Dummy3DPose(ThreeDPoseEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "Dummy3DPose"

    def _forward(self, poses_2d):
        return poses_2d


def example_1_full_pipeline(video_path: str):
    """Full pipeline from raw frames."""
    print("=== Full Pipeline ===")

    data_loader = ProcessManager(save_path="results.json")
    pipeline = EstimationPipe(
        DummyPreprocessor(), Dummy2DPose(), Dummy3DPose(), data_loader
    )

    # Set input data in dataloader
    raw_frames = torch.randn(5, 480, 640, 3)
    data_loader.set_input(raw_frames)

    # Run pipeline - no parameters needed!
    result = pipeline.forward()
    print(f"Final result: {result.shape}")


def example_2_from_2d_poses(video_path: str):
    """Load 2D poses and auto-continue from 3D lifting."""
    print("\n=== Auto-start from 2D Poses ===")

    data_loader = ProcessManager()
    data_loader.load_json("results.json")  # Has 2D poses

    # Pipeline auto-detects and starts from poselifting
    pipeline = EstimationPipe(
        DummyPreprocessor(), Dummy2DPose(), Dummy3DPose(), data_loader
    )

    # No input needed - uses stored 2D poses
    result = pipeline.forward()
    print(f"3D from stored 2D: {result.shape}")


def example_3_from_3d_poses(video_path: str):
    """Load 3D poses and auto-continue"""
    print("\n=== 3D Poses Input ===")

    data_loader = ProcessManager()
    data_loader.load_json("dataloader/results_3d.json")

    # Pipeline auto-detects and starts from poselifting
    pipeline = EstimationPipe(
        DummyPreprocessor(), Dummy2DPose(), Dummy3DPose(), data_loader
    )

    result = pipeline.forward()
    print(f"3D from manual 2D: {result.shape}")


def example_4_individual_stage(video_path: str):
    """Use dataloader's run_stage method directly."""
    print("\n=== Individual Stage Usage ===")

    data_loader = ProcessManager()

    # Add 2D poses
    poses_2d = torch.randn(2, 8, 17, 3)
    data_loader.handle(poses_2d, {"stage_name": "flatpose"})

    # Run only 3D lifting stage
    pose_3d_model = Dummy3DPose()
    result = data_loader.run_stage(pose_3d_model, "flatpose")
    print(f"Direct stage result: {result.shape}")


def example_5_from_video(video_path: str):
    """Load video and run pipeline with batch processing."""
    from estimation.preprocess.no_preprocess import NoPreprocess
    from estimation.pose2D.yolo_estimation import YOLOEstimation
    from estimation.pose3D.motionbert_estimation import MotionBERTEstimation

    print("\n=== Video Input with Batch Processing ===")

    # Use batch processing to handle large videos
    data_loader = ProcessManager(save_path="results-from-video.json", batch_size=32)
    data_loader.set_input_from_video(video_path, num_frames=120)

    print("Video prepared for batch processing.")

    pipeline = EstimationPipe(
        NoPreprocess(),
        YOLOEstimation("yolo11s-pose.pt"),
        MotionBERTEstimation(),
        data_loader,
    )

    result = pipeline.forward()
    print(f"Video result: {result.shape}")
    print(f"Processed {pipeline.processed_batches} batches")


def example6_motionbert_from_2d_poses(video_path: str):
    """Load 2D poses and run MotionBERT."""
    from estimation.preprocess.no_preprocess import NoPreprocess
    from estimation.pose2D.yolo_estimation import YOLOEstimation
    from estimation.pose3D.motionbert_estimation import MotionBERTEstimation

    print("\n=== MotionBERT from 2D Poses ===")

    data_loader = ProcessManager()
    data_loader.load_json("poseResearch/dataloader/results_flatpose.json")

    pipeline = EstimationPipe(
        NoPreprocess(),
        YOLOEstimation("yolo11s-pose.pt"),
        MotionBERTEstimation(),
        data_loader,
    )

    result = pipeline.forward()
    print(f"MotionBERT result: {result.shape}")


def example_7_yolo_bb_preprocess(video_path: str):
    """Run YOLO bounding box preprocess."""
    from estimation.preprocess.yolo_bb_preprocess import YOLOBoundingBoxPreprocess

    print("=== YOLO Bounding Box Preprocess ===")

    data_loader = ProcessManager(
        save_path="results-yolo11x-bb-preprocess_conv1_t1.json"
    )
    data_loader.set_input_from_video(video_path)

    pipeline = EstimationPipe(
        YOLOBoundingBoxPreprocess(
            model="yolo11l-pose.pt",
            video_path="yolo11x_bb_preprocess_conv1_t1.mp4",
        ),
        Dummy2DPose(),
        Dummy3DPose(),
        data_loader,
    )

    result = pipeline.forward()
    print(f"YOLO Bounding Box Preprocess result: {result.shape}")


def example_8_large_video_batch_processing(video_path: str):
    """Process a large video with small batch sizes to avoid memory issues."""
    from estimation.preprocess.no_preprocess import NoPreprocess
    from estimation.pose2D.yolo_estimation import YOLOEstimation
    from estimation.pose3D.motionbert_estimation import MotionBERTEstimation

    print("\n=== Large Video Batch Processing ===")

    # Use smaller batch size for very large videos or limited memory
    data_loader = ProcessManager(save_path="results-large-video.json", batch_size=100)
    data_loader.set_input_from_video(video_path, num_frames=400)

    print(f"Video prepared for batch processing: {data_loader.total_frames} frames")
    print(f"Batch size: {data_loader.batch_size}")

    pipeline = EstimationPipe(
        NoPreprocess(),
        YOLOEstimation("yolo11s-pose.pt"),
        MotionBERTEstimation(),
        data_loader,
    )

    result = pipeline.forward()
    print(f"Final result shape: {result.shape}")
    print(f"Total batches processed: {pipeline.processed_batches}")
    print(f"Frames per batch: {data_loader.batch_size}")
    print("Memory usage optimized for large videos!")


def example_9_detectron2_from_video(video_path: str):
    """Load video and run pipeline.
    To use this example, put a checkpoint into detectron2/model_zoo/checkpoints"""
    from estimation.preprocess.no_preprocess import NoPreprocess
    from estimation.pose2D.detectron2_estimation import Detectron2Estimation
    from estimation.pose3D.motionbert_estimation import MotionBERTEstimation

    print("\n=== Video Input ===")

    data_loader = ProcessManager(save_path="results-from-video.json")
    data_loader.set_input_from_video(video_path, num_frames=20)

    print("Data loading complete.")

    pipeline = EstimationPipe(
        NoPreprocess(),
        Detectron2Estimation(),
        MotionBERTEstimation(),
        data_loader,
    )

    result = pipeline.forward()
    print(f"Video result: {result.shape}")


def parse_args_and_examples():
    examples = {
        "1": ("Full Pipeline", "example_1_full_pipeline"),
        "2": ("From 2D Poses", "example_2_from_2d_poses"),
        "3": ("From 3D Poses", "example_3_from_3d_poses"),
        "4": ("Individual Stage", "example_4_individual_stage"),
        "5": ("From Video", "example_5_from_video"),
        "6": ("MotionBERT from 2D Poses", "example6_motionbert_from_2d_poses"),
        "7": ("YOLO Bounding Box Preprocess", "example_7_yolo_bb_preprocess"),
        "8": ("Large Video Batch Processing", "example_8_large_video_batch_processing"),
        "9": ("Detectron2 from Video", "example_9_detectron2_from_video"),
    }

    parser = argparse.ArgumentParser(
        description="Run poseResearch example usage scripts by number."
    )
    parser.add_argument(
        "example",
        type=str,
        choices=examples.keys(),
        help="Example number to run: "
        + ", ".join(f"{k}: {v[0]}" for k, v in examples.items()),
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path of input video to process",
        default="fem1_t1_preview.mp4",
    )
    args = parser.parse_args()
    return args, examples


if __name__ == "__main__":
    args, examples = parse_args_and_examples()

    # Import all example functions into the local namespace
    local_vars = globals()
    # If the functions are not in globals (e.g. if this is run as a script), use locals()
    if "example_1_full_pipeline" not in local_vars:
        local_vars = locals()

    func_name = examples[args.example][1]
    if func_name in local_vars:
        local_vars[func_name](args.video)
    else:
        print(f"Function {func_name} not found.")
        sys.exit(1)
