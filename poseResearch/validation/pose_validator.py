from typing import List, Optional, Tuple
import numpy as np
import cv2
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import DataLoader, StageName


class PoseValidator:
    """
    Simple pose validator that computes similarity between 3D poses and 2D ground truth poses.
    Accepts two DataLoaders: one for ground truth 2D poses and one for 3D poses.
    """

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

    def validate(
        self,
        gt_data_loader: DataLoader,
        poses_3d_data_loader: DataLoader,
        gt_stage: StageName = "flatpose",
        poses_3d_stage: StageName = "poselifting",
    ) -> float:
        """
        Validate 3D poses against 2D ground truth poses from DataLoaders.

        Args:
            gt_data_loader: DataLoader containing ground truth 2D poses
            poses_3d_data_loader: DataLoader containing 3D poses
            gt_stage: Stage name in gt_data_loader containing 2D poses
            poses_3d_stage: Stage name in poses_3d_data_loader containing 3D poses

        Returns:
            Average similarity score across all comparisons
        """
        # Get data from DataLoaders
        gt_poses = gt_data_loader.get_tensor(gt_stage)
        poses_3d = poses_3d_data_loader.get_tensor(poses_3d_stage)

        if gt_poses is None:
            raise ValueError(f"No ground truth data found for stage '{gt_stage}'")
        if poses_3d is None:
            raise ValueError(f"No 3D poses data found for stage '{poses_3d_stage}'")

        return self._validate_poses(poses_3d, gt_poses)

    def validate_batch(
        self,
        gt_data_loaders: List[DataLoader],
        poses_3d_data_loaders: List[DataLoader],
        gt_stage: StageName = "flatpose",
        poses_3d_stage: StageName = "poselifting",
    ) -> List[float]:
        """
        Validate a batch of pose sequences from multiple DataLoaders.

        Args:
            gt_data_loaders: List of DataLoaders containing ground truth 2D poses
            poses_3d_data_loaders: List of DataLoaders containing 3D poses
            gt_stage: Stage name containing 2D poses
            poses_3d_stage: Stage name containing 3D poses

        Returns:
            List of similarity scores for each sequence pair
        """
        if len(gt_data_loaders) != len(poses_3d_data_loaders):
            raise ValueError(
                "Number of ground truth and 3D pose DataLoaders must match"
            )

        scores = []
        for gt_loader, poses_loader in zip(gt_data_loaders, poses_3d_data_loaders):
            score = self.validate(gt_loader, poses_loader, gt_stage, poses_3d_stage)
            scores.append(score)

        return scores

    def _validate_poses(
        self, poses_3d: torch.Tensor, gt_poses_2d: torch.Tensor
    ) -> float:
        """
        Core validation logic between 3D poses and 2D ground truth poses.

        Args:
            poses_3d: 3D poses tensor of shape (num_people, num_frames, num_keypoints, 3)
            gt_poses_2d: 2D ground truth poses of shape (num_people, num_frames, num_keypoints, 3)
                         where the last dimension contains [x, y, confidence]

        Returns:
            Average similarity score across all comparisons
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

        similarities = []
        num_frames = min(poses_3d.shape[1], gt_poses_2d.shape[1])

        # Process each frame
        for frame_idx in range(num_frames):
            frame_3d = poses_3d[:, frame_idx]  # (num_people, num_keypoints, 3)
            frame_2d_gt = gt_poses_2d[:, frame_idx]  # (num_people, num_keypoints, 3)

            frame_similarity = self._compute_frame_similarity(frame_3d, frame_2d_gt)
            print(frame_similarity)
            if frame_similarity is not None:
                similarities.append(frame_similarity)

        return np.mean(similarities) if similarities else 0.0

    def _compute_frame_similarity(
        self, poses_3d_frame: np.ndarray, poses_2d_gt_frame: np.ndarray
    ) -> Optional[float]:
        """
        Compute similarity for a single frame using Hungarian algorithm for optimal assignment.
        """
        num_people_3d = poses_3d_frame.shape[0]
        num_people_2d = poses_2d_gt_frame.shape[0]

        if num_people_3d == 0 or num_people_2d == 0:
            return None

        # Compute similarity matrix
        similarity_matrix = np.zeros((num_people_3d, num_people_2d))

        for i, pose_3d in enumerate(poses_3d_frame):
            for j, pose_2d_gt in enumerate(poses_2d_gt_frame):
                similarity = self._compute_pose_similarity(pose_3d, pose_2d_gt)
                similarity_matrix[i, j] = similarity

        # Optimal assignment using Hungarian algorithm
        if similarity_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
            assigned_similarities = similarity_matrix[row_indices, col_indices]
            return np.mean(assigned_similarities)

        return None

    def _compute_pose_similarity(
        self, pose_3d: np.ndarray, pose_2d_gt: np.ndarray
    ) -> float:
        """
        Compute similarity between a single 3D pose and 2D ground truth pose using PnP projection.
        """
        try:
            # Extract 2D coordinates and confidence
            keypoints_2d = pose_2d_gt[:, :2]  # (num_keypoints, 2)
            confidence = pose_2d_gt[:, 2]  # (num_keypoints,)
            print(confidence)

            # Filter by confidence threshold
            valid_mask = confidence > self.confidence_threshold

            if valid_mask.sum() < 4:
                return 0.0  # Need at least 4 points for PnP

            # Get valid keypoints
            valid_keypoints_2d = keypoints_2d[valid_mask].astype(np.float32)
            valid_keypoints_3d = pose_3d[valid_mask].astype(np.float32)

            # Project 3D pose to 2D using PnP
            success, rvec, tvec, _ = cv2.solvePnPRansac(
                valid_keypoints_3d,
                valid_keypoints_2d,
                self.camera_matrix,
                self.dist_coeffs,
            )

            if not success:
                return 0.0

            # Project 3D points to 2D
            projected_points, _ = cv2.projectPoints(
                valid_keypoints_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)

            # Compute cosine similarity
            cos_sim_matrix = cosine_similarity(projected_points, valid_keypoints_2d)
            cos_sim = np.diag(cos_sim_matrix)

            # Return mean similarity (handle NaN)
            cos_sim = np.nan_to_num(cos_sim, nan=0.0)
            return np.mean(cos_sim)

        except Exception:
            return 0.0
