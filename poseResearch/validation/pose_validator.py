from typing import List, Tuple
import numpy as np
import cv2
import torch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.process_manager import ProcessManager, StageName


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
    ):
        """
        Initialize the pose validator.

        Args:
            confidence_threshold: Minimum confidence for 2D keypoints to be considered valid
            image_size: Image size (width, height) for camera projection
        """
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.focal_length = image_size[0]  # Use image width as focal length

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

    def validate(
        self,
        gt_data_loader: ProcessManager,
        poses_3d_data_loader: ProcessManager,
        gt_stage: StageName = "flatpose",
        poses_3d_stage: StageName = "poselifting",
        use_real_world_scale: bool = True,
    ) -> float:
        """
        Validate 3D poses against 2D ground truth poses from DataLoaders using MPJPE.

        Args:
            gt_data_loader: ProcessManager containing ground truth 2D poses
            poses_3d_data_loader: ProcessManager containing 3D poses
            gt_stage: Stage name in gt_data_loader containing 2D poses
            poses_3d_stage: Stage name in poses_3d_data_loader containing 3D poses
            use_real_world_scale: If True, scale MPJPE to real-world millimeters

        Returns:
            Average MPJPE across all comparisons (in pixels or millimeters)
        """
        # Get data from DataLoaders
        gt_poses = gt_data_loader.get_tensor(gt_stage)
        poses_3d = poses_3d_data_loader.get_tensor(poses_3d_stage)

        if gt_poses is None:
            raise ValueError(f"No ground truth data found for stage '{gt_stage}'")
        if poses_3d is None:
            raise ValueError(f"No 3D poses data found for stage '{poses_3d_stage}'")

        return self._validate_poses(poses_3d, gt_poses, use_real_world_scale)

    def _validate_poses(
        self,
        poses_3d: torch.Tensor,
        gt_poses_2d: torch.Tensor,
        use_real_world_scale: bool = False,
    ) -> float:
        """
        Core validation logic using MPJPE calculation.

        Args:
            poses_3d: 3D poses tensor of shape (num_people, num_frames, num_keypoints, 3)
            gt_poses_2d: 2D ground truth poses of shape (num_people, num_frames, num_keypoints, 3)
                         where the last dimension contains [x, y, confidence]
            use_real_world_scale: If True, scale MPJPE to real-world millimeters

        Returns:
            Average MPJPE across all comparisons (in pixels or millimeters)
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
        num_frames = min(poses_3d.shape[1], gt_poses_2d.shape[1])

        # Process each frame
        for frame_idx in range(num_frames):
            frame_3d = poses_3d[:, frame_idx]  # (num_people, num_keypoints, 3)
            frame_2d_gt = gt_poses_2d[:, frame_idx]  # (num_people, num_keypoints, 3)

            # Extract 2D coordinates (ignore confidence for now)
            frame_2d_coords = frame_2d_gt[:, :, :2]  # (num_people, num_keypoints, 2)

            # Filter by confidence if needed
            confidence = frame_2d_gt[:, :, 2]  # (num_people, num_keypoints)
            # valid_people = []
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

                # Calculate MPJPE for this frame
                if use_real_world_scale:
                    frame_mpjpe = self.real_mpjpe(valid_3d, valid_2d)
                else:
                    frame_mpjpe = self.mpjpe(valid_3d, valid_2d)
                print(frame_mpjpe)
                mpjpe_scores.append(frame_mpjpe)

        return np.mean(mpjpe_scores) if mpjpe_scores else float("inf")
