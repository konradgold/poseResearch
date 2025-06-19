#!/usr/bin/env python3
"""
Video Pose Validation Script

This script loads a video and ground truth keypoints, uses YOLO pose model 
to generate 2D keypoint predictions, and validates using similarity metrics
between projected ground truth and YOLO predictions.
"""

import argparse
import pickle
import cv2
import os
import json
import glob
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict
from ultralytics import YOLO
from pose_validation import PoseValidation
import yaml


class MediaLoader:
    """Handles video loading and image directory loading"""

    def __init__(self, media_path: str):
        self.media_path = media_path
        self.is_directory = os.path.isdir(media_path)

        if self.is_directory:
            # Load image files from directory
            self.image_extensions = [
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.bmp",
                "*.tiff",
                "*.tif",
            ]
            self.image_files = []
            for ext in self.image_extensions:
                self.image_files.extend(glob.glob(os.path.join(media_path, ext)))
                self.image_files.extend(
                    glob.glob(os.path.join(media_path, ext.upper()))
                )
            self.image_files.sort()  # Ensure consistent ordering

            if not self.image_files:
                raise ValueError(f"No image files found in directory: {media_path}")
            print(f"Found {len(self.image_files)} images in directory")
        else:
            # Video file
            self.cap = cv2.VideoCapture(media_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {media_path}")

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Number of frames: {self.total_frames}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def get_frames(self, sample_rate: int = 1) -> List[np.ndarray]:
        """Extract frames with optional sampling and return frame names/indices"""
        frames = []

        if self.is_directory:
            # Load images from directory
            for i, img_path in enumerate(self.image_files):
                if i % sample_rate == 0:
                    frame = cv2.imread(img_path)
                    if frame is not None:
                        frames.append(frame)
        else:
            # Extract from video
            frame_idx = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    frames.append(frame)
                frame_idx += 1

            self.cap.release()

        return frames

    def __del__(self):
        if hasattr(self, "cap") and not self.is_directory and self.cap.isOpened():
            self.cap.release()


class GroundTruthLoader:
    """Handles loading of ground truth 3D keypoints"""

    @staticmethod
    def load_from_pickle(pickle_path: str) -> List[np.ndarray]:
        """Load ground truth keypoints from pickle file"""
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        # Convert to list of numpy arrays if needed
        if isinstance(data, np.ndarray):
            if data.ndim == 3:  # (frames, joints, 3)
                return [data[i] for i in range(data.shape[0])]
            else:
                return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unsupported ground truth data type: {type(data)}")

    @staticmethod
    def load_from_json_directory(json_dir: str) -> List[np.ndarray]:
        """Load ground truth keypoints from JSON files matching frame names"""
        ground_truth_poses = []

        # Get all JSON files in the directory and sort them lexically
        json_files = sorted([f for f in glob.glob(os.path.join(json_dir, "*.json"))])
        # Map frame_names (without extension) to JSON files
        # json_file_map = {
        #     os.path.splitext(os.path.basename(f))[0]: f for f in json_files
        # }

        # Iterate through sorted json_files directly
        for json_path in json_files:
            with open(json_path, "r") as jf:
                json_data = json.load(jf)
            pose = GroundTruthLoader._extract_pose_from_json(json_data)
            ground_truth_poses.append(pose)

        return ground_truth_poses

    @staticmethod
    def _extract_pose_from_json(json_data: Dict) -> np.ndarray:
        # Only extract 'pose_keypoints_2d' from the first person, ignore others
        if "people" in json_data and len(json_data["people"]) > 0:
            keypoints_all = []
            for person in json_data["people"]:
                if "pose_keypoints_2d" in person:
                    kpts_2d = person["pose_keypoints_2d"]
                    # Extract x, y only, ignore confidence
                    keypoints = []
                    for i in range(0, len(kpts_2d), 3):
                        keypoints.extend([kpts_2d[i], kpts_2d[i + 1], 0.0])  # z=0
                    keypoints_all.append(
                        np.array(keypoints, dtype=np.float32).reshape(-1, 3)
                    )
            if keypoints_all:
                return np.stack(
                    keypoints_all, axis=0
                )  # shape: (num_people, num_joints, 3)
        else:
            raise ValueError("No pose_keypoints_2d found in person data")


class YoloPoseInference:
    """Handles YOLO pose model inference"""

    def __init__(self, model_path: str = "yolo11n-pose.pt"):
        self.model = YOLO(model_path)

    def predict_frames(self, frames: List[np.ndarray], batch_size: int = 8) -> List:
        """Run YOLO pose inference on frames"""
        all_results = []

        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            results = self.model(batch_frames)
            all_results.extend(results)

        return all_results

    def extract_keypoints(self, results: List) -> List:
        """Extract keypoints from YOLO results"""
        keypoints_list = []

        for result in results:
            if result.keypoints is not None:
                keypoints_list.append(result.keypoints)
            else:
                # If no keypoints detected, create empty placeholder
                keypoints_list.append(None)

        return keypoints_list


def create_timestamped_directory(base_name: str = "pose_validation") -> str:
    """Create directory with timestamp in name"""
    now = datetime.now()
    timestamp = now.strftime("%H%M")  # Hour and minute
    dir_name = f"{base_name}_{timestamp}"

    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def align_sequences(ground_truth: List, yolo_keypoints: List) -> Tuple[List, List]:
    """Align ground truth and YOLO sequences"""
    # Both sequences should already be aligned since they're based on the same frames
    min_length = min(len(ground_truth), len(yolo_keypoints))
    return ground_truth[:min_length], yolo_keypoints[:min_length]


class CustomPoseValidation(PoseValidation):
    """Extended PoseValidation for YOLO keypoints"""

    def preprocess_poses(self, yolo_keypoints, ground_truth_3d):  # type: ignore
        """Preprocess YOLO keypoints and ground truth for validation"""
        if yolo_keypoints is None:
            # Return empty structures if no keypoints detected
            return [], np.array([]).reshape(0, 3)

        # YOLO keypoints come as a list, ground truth is a single 3D pose
        return [yolo_keypoints], np.array([ground_truth_3d])


def main():
    parser = argparse.ArgumentParser(
        description="Validate YOLO pose predictions against ground truth"
    )
    parser.add_argument(
        "--media", type=str, required=True, help="Path to video file or image directory"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth (.pkl file or JSON directory)",
    )
    parser.add_argument(
        "--model", type=str, default="yolo11n-pose.pt", help="Path to YOLO pose model"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=1, help="Frame sampling rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for YOLO inference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: timestamped)",
    )

    args = parser.parse_args()

    print(f"Loading media: {args.media}")
    print(f"Loading ground truth: {args.ground_truth}")
    print(f"Using YOLO model: {args.model}")

    # Create output directory
    if args.output_dir is None:
        output_dir = create_timestamped_directory()
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load media and extract frames
    media_loader = MediaLoader(args.media)
    frames = media_loader.get_frames(sample_rate=args.sample_rate)
    print(f"Extracted {len(frames)} frames")

    # Load ground truth
    if args.ground_truth.endswith(".pkl"):
        ground_truth = GroundTruthLoader.load_from_pickle(args.ground_truth)
        print(f"Loaded {len(ground_truth)} ground truth poses from pickle")
    else:
        # Assume JSON directory
        ground_truth = GroundTruthLoader.load_from_json_directory(args.ground_truth)
        print(f"Loaded {len(ground_truth)} ground truth poses from JSON files")

    # Run YOLO pose inference
    yolo_inference = YoloPoseInference(args.model)
    results = yolo_inference.predict_frames(frames, batch_size=args.batch_size)
    yolo_keypoints = yolo_inference.extract_keypoints(results)
    print(f"Generated {len(yolo_keypoints)} YOLO pose predictions")

    # Align sequences
    aligned_gt, aligned_yolo = align_sequences(ground_truth, yolo_keypoints)
    print(
        f"Aligned sequences: {len(aligned_gt)} ground truth, {len(aligned_yolo)} YOLO predictions"
    )

    # Validate poses
    validator = CustomPoseValidation(
        ground_truth=aligned_yolo,  # YOLO keypoints (2D)
        poses=aligned_gt,  # Ground truth poses (3D)
        output_mode=f"PATH:{output_dir}/validation_results.txt",
    )

    mean_similarity = validator.average_similarity()
    print(f"Mean similarity: {mean_similarity:.4f}")

    # Save results
    results_file = os.path.join(output_dir, "mean_similarity.yml")
    results_data = {
        "mean_similarity": float(f"{mean_similarity:.6f}"),
        "media": args.media,
        "ground_truth": args.ground_truth,
        "model": args.model,
        "sample_rate": args.sample_rate,
        "aligned_sequences": len(aligned_gt),
        "media_type": "directory" if media_loader.is_directory else "video",
    }
    with open(results_file, "w") as f:
        yaml.dump(results_data, f, default_flow_style=False)

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
