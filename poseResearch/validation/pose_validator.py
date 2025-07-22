from typing import Tuple, Dict, Optional
import numpy as np
import cv2
import torch
import csv
import os

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.process_manager import ProcessManager, StageName
from visualizer.skeleton_config import AnatomicalSkeletonConfig


class PoseValidator:
    """
    Simple pose validator that computes MPJPE between 3D poses and 2D ground truth poses.
    Accepts two DataLoaders: one for ground truth 2D poses and one for 3D poses.
    """

    # Constants for real-world measurements
    AVERAGE_HUMAN_HEIGHT_MM = 1700.0  # Average human height in millimeters
    ROOT_TO_SPINE_RATIO = (
        0.16  # Root to spine distance as proportion of total body height
    )
    ROOT_KEYPOINT_INDEX = 0  # Index of root keypoint
    SPINE_KEYPOINT_INDEX = 7  # Index of spine keypoint

    def __init__(
        self,
        confidence_threshold: float = 0.4,
        image_size: Tuple[int, int] = (640, 480),
        skeleton_config: Optional[AnatomicalSkeletonConfig] = None,
    ):
        """
        Initialize the pose validator.

        Args:
            confidence_threshold: Minimum confidence for 2D keypoints to be considered valid
            image_size: Image size (width, height) for camera projection
            skeleton_config: Skeleton configuration for joint names and structure
        """
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.focal_length = image_size[0]  # Use image width as focal length

        # Initialize skeleton configuration
        if skeleton_config is None:
            skeleton_config = AnatomicalSkeletonConfig()
        self.skeleton_config = skeleton_config
        self.joint_names = skeleton_config.get_keypoint_names()

        # Camera matrix for 3D to 2D projection
        self.camera_matrix = np.array(
            [
                [self.focal_length, 0, image_size[0] / 2],
                [0, self.focal_length, image_size[1] / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        # No distortion
        self.dist_coeffs = np.zeros((4, 1))

        # Calculate expected real-world root-to-spine distance
        self.expected_root_to_spine_distance_mm = (
            self.AVERAGE_HUMAN_HEIGHT_MM * self.ROOT_TO_SPINE_RATIO
        )

    def project_to_plane(
        self, predicted_3d: np.ndarray, target_2d: np.ndarray
    ) -> np.ndarray:
        """
        Project 3D poses to 2D plane using PnP algorithm, similar to torch_project_to_plane.

        Args:
            predicted_3d: 3D poses of shape (num_poses, num_keypoints, 3)
            target_2d: 2D target poses of shape (num_poses, num_keypoints, 2)

        Returns:
            Projected 2D points of shape (num_poses, num_keypoints, 2)
        """
        assert len(predicted_3d.shape) == 3
        assert len(target_2d.shape) == 3

        assert predicted_3d.shape[0] == target_2d.shape[0]
        assert predicted_3d.shape[1] == target_2d.shape[1]
        assert predicted_3d.shape[2] == target_2d.shape[2] + 1

        out = np.zeros(target_2d.shape)

        for i, keypoints_3d in enumerate(predicted_3d):
            # Ensure correct dtype/shape
            keypoints_3d_np = keypoints_3d.astype(np.float64)
            target_np = target_2d[i].astype(np.float64)

            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    keypoints_3d_np, target_np, self.camera_matrix, self.dist_coeffs
                )

                if not success:
                    continue  # Skip this pair if PnP fails

                projected_points, _ = cv2.projectPoints(
                    keypoints_3d_np, rvec, tvec, self.camera_matrix, self.dist_coeffs
                )

                out[i] = projected_points.reshape(-1, 2)

            except Exception:
                continue  # Skip this pair if projection fails

        return out

    def _calculate_scale_factor(self, predicted_3d: np.ndarray) -> float:
        """
        Calculate the scale factor to convert from model units to real-world millimeters.

        Args:
            predicted_3d: 3D poses of shape (num_poses, num_keypoints, 3)

        Returns:
            Scale factor to convert model units to millimeters
        """
        root_to_spine_distances = []

        for pose in predicted_3d:
            root_point = pose[self.ROOT_KEYPOINT_INDEX]
            spine_point = pose[self.SPINE_KEYPOINT_INDEX]

            # Calculate Euclidean distance between root and spine
            distance = np.linalg.norm(spine_point - root_point)
            if distance > 0:  # Avoid division by zero
                root_to_spine_distances.append(distance)

        if not root_to_spine_distances:
            return 1.0  # Default scale factor if no valid distances

        # Use median distance to avoid outliers
        median_measured_distance = np.median(root_to_spine_distances)

        # Calculate scale factor: real_distance / measured_distance
        scale_factor = (
            self.expected_root_to_spine_distance_mm / median_measured_distance
        )

        return scale_factor

    def mpjpe(self, predicted_3d: np.ndarray, target_2d: np.ndarray) -> float:
        """
        Mean per-joint position error (i.e. mean Euclidean distance),
        following the same approach as the marked loss_mpjpe function.
        """
        predicted_2d = self.project_to_plane(predicted_3d, target_2d)
        assert predicted_2d.shape == target_2d.shape
        return np.mean(
            np.linalg.norm(predicted_2d - target_2d, axis=len(target_2d.shape) - 1)
        )

    def real_mpjpe(self, predicted_3d: np.ndarray, target_2d: np.ndarray) -> float:
        """
        Mean per-joint position error scaled to real-world measurements in millimeters.

        This method calculates the scale factor based on the root-to-spine distance
        and applies it to convert the MPJPE to real-world millimeters.

        Args:
            predicted_3d: 3D poses of shape (num_poses, num_keypoints, 3)
            target_2d: 2D target poses of shape (num_poses, num_keypoints, 2)

        Returns:
            MPJPE in millimeters based on real human proportions
        """
        # Calculate scale factor from root-to-spine distance
        scale_factor = self._calculate_scale_factor(predicted_3d)

        # Project 3D poses to 2D
        predicted_2d = self.project_to_plane(predicted_3d, target_2d)
        assert predicted_2d.shape == target_2d.shape

        # Calculate MPJPE and scale to real-world measurements
        mpjpe_model_units = np.mean(
            np.linalg.norm(predicted_2d - target_2d, axis=len(target_2d.shape) - 1)
        )

        # Convert to millimeters
        real_mpjpe_mm = mpjpe_model_units * scale_factor

        return real_mpjpe_mm

    def mpjpe_per_joint(
        self, predicted_3d: np.ndarray, target_2d: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate MPJPE per joint (keypoint).

        Args:
            predicted_3d: 3D poses of shape (num_poses, num_keypoints, 3)
            target_2d: 2D target poses of shape (num_poses, num_keypoints, 2)

        Returns:
            Dictionary mapping joint names to their MPJPE values
        """
        predicted_2d = self.project_to_plane(predicted_3d, target_2d)
        assert predicted_2d.shape == target_2d.shape

        per_joint_errors = {}

        # Calculate error for each joint
        for joint_idx, joint_name in enumerate(self.joint_names):
            if joint_idx < predicted_2d.shape[1]:  # Ensure joint exists in data
                joint_errors = np.linalg.norm(
                    predicted_2d[:, joint_idx, :] - target_2d[:, joint_idx, :], axis=1
                )
                per_joint_errors[joint_name] = np.mean(joint_errors)
            else:
                per_joint_errors[joint_name] = float("inf")

        return per_joint_errors

    def real_mpjpe_per_joint(
        self, predicted_3d: np.ndarray, target_2d: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate real-world scaled MPJPE per joint (keypoint) in millimeters.

        Args:
            predicted_3d: 3D poses of shape (num_poses, num_keypoints, 3)
            target_2d: 2D target poses of shape (num_poses, num_keypoints, 2)

        Returns:
            Dictionary mapping joint names to their real-world MPJPE values in mm
        """
        # Calculate scale factor from root-to-spine distance
        scale_factor = self._calculate_scale_factor(predicted_3d)

        # Get per-joint errors in model units
        per_joint_errors = self.mpjpe_per_joint(predicted_3d, target_2d)

        # Scale to real-world measurements
        real_per_joint_errors = {
            joint_name: error * scale_factor
            for joint_name, error in per_joint_errors.items()
        }

        return real_per_joint_errors

    def save_results_to_csv(
        self,
        results: Dict[str, any],
        output_path: str = "pose_validation_results.csv",
    ) -> None:
        """
        Save validation results to a CSV file with each frame as a separate row.

        Args:
            results: Dictionary containing validation results
            output_path: Path to save the CSV file
        """
        # Ensure output directory exists
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        # Prepare header fields
        header_fields = [
            "frame_idx",
            "overall_mpjpe",
            "overall_real_mpjpe_mm",
            "confidence_threshold",
        ]

        # Add per-joint MPJPE fields
        for joint_name in self.joint_names:
            header_fields.extend([f"mpjpe_{joint_name}", f"real_mpjpe_mm_{joint_name}"])

        # Write to CSV (always replace, never append)
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header_fields)
            writer.writeheader()

            # Write summary row
            summary_row = {
                "frame_idx": "SUMMARY",
                "overall_mpjpe": results.get("overall_mpjpe", 0.0),
                "overall_real_mpjpe_mm": results.get("overall_real_mpjpe_mm", 0.0),
                "confidence_threshold": self.confidence_threshold,
            }

            # Add per-joint summary data
            per_joint_mpjpe = results.get("per_joint_mpjpe", {})
            per_joint_real_mpjpe = results.get("per_joint_real_mpjpe_mm", {})

            for joint_name in self.joint_names:
                summary_row[f"mpjpe_{joint_name}"] = per_joint_mpjpe.get(
                    joint_name, 0.0
                )
                summary_row[f"real_mpjpe_mm_{joint_name}"] = per_joint_real_mpjpe.get(
                    joint_name, 0.0
                )

            writer.writerow(summary_row)

            # Write individual frame data
            frame_data = results.get("frame_data", [])
            for frame_idx, frame_result in enumerate(frame_data):
                frame_row = {
                    "frame_idx": frame_idx,
                    "overall_mpjpe": frame_result.get("overall_mpjpe", 0.0),
                    "overall_real_mpjpe_mm": frame_result.get(
                        "overall_real_mpjpe_mm", 0.0
                    ),
                    "confidence_threshold": self.confidence_threshold,
                }

                # Add per-joint frame data
                frame_per_joint_mpjpe = frame_result.get("per_joint_mpjpe", {})
                frame_per_joint_real_mpjpe = frame_result.get(
                    "per_joint_real_mpjpe_mm", {}
                )

                for joint_name in self.joint_names:
                    frame_row[f"mpjpe_{joint_name}"] = frame_per_joint_mpjpe.get(
                        joint_name, 0.0
                    )
                    frame_row[f"real_mpjpe_mm_{joint_name}"] = (
                        frame_per_joint_real_mpjpe.get(joint_name, 0.0)
                    )

                writer.writerow(frame_row)

        print(f"Results saved to {output_path} with {len(frame_data)} frame rows")

    def validate(
        self,
        gt_data_loader: ProcessManager,
        poses_3d_data_loader: ProcessManager,
        gt_name: str,
        poses_3d_name: str,
        use_real_world_scale: bool = True,
        save_to_csv: bool = True,
    ) -> Dict[str, any]:
        """
        Validate 3D poses against 2D ground truth poses from DataLoaders using MPJPE.

        Args:
            gt_data_loader: ProcessManager containing ground truth 2D poses
            poses_3d_data_loader: ProcessManager containing 3D poses
            gt_name: Name of the ground truth data
            poses_3d_name: Name of the 3D poses data
            use_real_world_scale: If True, scale MPJPE to real-world millimeters
            save_to_csv: Whether to save results to CSV file

        Returns:
            Dictionary containing validation results including per-joint MPJPE
        """
        # Get data from DataLoaders
        gt_poses = gt_data_loader.get_tensor("flatpose")
        poses_3d = poses_3d_data_loader.get_tensor("poselifting")

        if gt_poses is None:
            raise ValueError(f"No ground truth data found for stage 'flatpose'")
        if poses_3d is None:
            raise ValueError(f"No 3D poses data found for stage 'poselifting'")

        results = self._validate_poses(poses_3d, gt_poses, use_real_world_scale)

        if save_to_csv:
            pr_dir = Path(__file__).parent
            csv_output_path = (
                pr_dir
                / "validation-results"
                / f"validation-gt-{gt_name}--3d-{poses_3d_name}.csv"
            )
            self.save_results_to_csv(results, csv_output_path)

        return results

    def _validate_poses(
        self,
        poses_3d: torch.Tensor,
        gt_poses_2d: torch.Tensor,
        use_real_world_scale: bool = False,
    ) -> Dict[str, any]:
        """
        Core validation logic using MPJPE calculation with detailed per-joint analysis.

        Args:
            poses_3d: 3D poses tensor of shape (num_people, num_frames, num_keypoints, 3)
            gt_poses_2d: 2D ground truth poses of shape (num_people, num_frames, num_keypoints, 3)
                         where the last dimension contains [x, y, confidence]
            use_real_world_scale: If True, scale MPJPE to real-world millimeters

        Returns:
            Dictionary containing overall MPJPE, per-joint MPJPE, and metadata
        """
        # Convert to numpy
        if isinstance(poses_3d, torch.Tensor):
            poses_3d = poses_3d.detach().cpu().numpy()
        if isinstance(gt_poses_2d, torch.Tensor):
            gt_poses_2d = gt_poses_2d.detach().cpu().numpy()

        # Validate shapes
        if poses_3d.ndim != 4 or poses_3d.shape[3] != 3:
            raise ValueError(
                f"Expected 3D poses shape (people, frames, keypoints, 3), got {poses_3d.shape}"
            )
        if gt_poses_2d.ndim != 4 or gt_poses_2d.shape[3] != 3:
            raise ValueError(
                f"Expected 2D GT poses shape (people, frames, keypoints, 3), got {gt_poses_2d.shape}"
            )

        mpjpe_scores = []
        all_per_joint_errors = []
        frame_data = []
        num_frames = min(poses_3d.shape[1], gt_poses_2d.shape[1])
        frames_processed = 0

        # Process each frame
        for frame_idx in range(num_frames):
            frame_3d = poses_3d[:, frame_idx]  # (num_people, num_keypoints, 3)
            frame_2d_gt = gt_poses_2d[:, frame_idx]  # (num_people, num_keypoints, 3)

            # Extract 2D coordinates (ignore confidence for now)
            frame_2d_coords = frame_2d_gt[:, :, :2]  # (num_people, num_keypoints, 2)

            # Filter by confidence if needed
            confidence = frame_2d_gt[:, :, 2]  # (num_people, num_keypoints)
            valid_3d = []
            valid_2d = []

            for person_idx in range(min(frame_3d.shape[0], frame_2d_coords.shape[0])):
                person_confidence = confidence[person_idx]
                valid_mask = person_confidence > self.confidence_threshold

                if valid_mask.sum() >= 4:  # Need at least 4 points for PnP
                    valid_3d.append(frame_3d[person_idx])
                    valid_2d.append(frame_2d_coords[person_idx])

            if len(valid_3d) > 0:
                valid_3d = np.array(valid_3d)
                valid_2d = np.array(valid_2d)

                # Calculate overall MPJPE for this frame
                if use_real_world_scale:
                    frame_mpjpe = self.real_mpjpe(valid_3d, valid_2d)
                    frame_per_joint_errors = self.real_mpjpe_per_joint(
                        valid_3d, valid_2d
                    )
                else:
                    frame_mpjpe = self.mpjpe(valid_3d, valid_2d)
                    frame_per_joint_errors = self.mpjpe_per_joint(valid_3d, valid_2d)

                print(f"Frame {frame_idx}: Overall MPJPE = {frame_mpjpe:.3f}")
                mpjpe_scores.append(frame_mpjpe)
                all_per_joint_errors.append(frame_per_joint_errors)
                frames_processed += 1

                # Store frame data for CSV output
                frame_result = {
                    "overall_mpjpe": frame_mpjpe,
                    "per_joint_mpjpe": frame_per_joint_errors,
                }

                if use_real_world_scale:
                    frame_result["overall_real_mpjpe_mm"] = frame_mpjpe
                    frame_result["per_joint_real_mpjpe_mm"] = frame_per_joint_errors

                frame_data.append(frame_result)

        # Calculate overall statistics
        overall_mpjpe = np.mean(mpjpe_scores) if mpjpe_scores else float("inf")

        # Calculate average per-joint errors across all frames
        if all_per_joint_errors:
            avg_per_joint_errors = {}
            for joint_name in self.joint_names:
                joint_errors = [
                    errors.get(joint_name, float("inf"))
                    for errors in all_per_joint_errors
                ]
                valid_errors = [e for e in joint_errors if e != float("inf")]
                avg_per_joint_errors[joint_name] = (
                    np.mean(valid_errors) if valid_errors else float("inf")
                )
        else:
            avg_per_joint_errors = {
                joint_name: float("inf") for joint_name in self.joint_names
            }

            # Prepare results dictionary
        results = {
            "overall_mpjpe": overall_mpjpe,
            "per_joint_mpjpe": avg_per_joint_errors,
            "num_frames_processed": frames_processed,
            "total_frames": num_frames,
            "frame_data": frame_data,
        }

        # Add real-world scale results if applicable
        if use_real_world_scale:
            results["overall_real_mpjpe_mm"] = overall_mpjpe
            results["per_joint_real_mpjpe_mm"] = avg_per_joint_errors

        # Print summary
        print("\n=== Pose Validation Results ===")
        print(
            f"Overall MPJPE: {overall_mpjpe:.3f} {'mm' if use_real_world_scale else 'pixels'}"
        )
        print(f"Frames processed: {frames_processed}/{num_frames}")
        print("\nPer-Joint MPJPE:")
        for joint_name, error in avg_per_joint_errors.items():
            if error != float("inf"):
                print(
                    f"  {joint_name:15}: {error:.3f} {'mm' if use_real_world_scale else 'pixels'}"
                )

        return results
