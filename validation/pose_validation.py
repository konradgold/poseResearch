import os
from typing import List
import numpy as np
import cv2
import torch
from scipy.optimize import linear_sum_assignment

class PoseValidation:
    confidence_threshold = 0.4
    def __init__(self, ground_truth, poses, output_mode: str):
        self.ground_truth = ground_truth
        self.poses = poses
        self.output_mode = output_mode

        if self.output_mode.startswith("PATH:"):
            file_path = self.output_mode[len("PATH:"):]
            file_path = file_path.strip()
            if file_path and not os.path.exists(file_path):
                # Create the file and its parent directories if needed
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def pose_keypoint_similarity(self, keypoints, poses) -> float:
        m_error = np.array((len(keypoints), len(poses)))
        for i,keypoints2d in enumerate(keypoints):
            keypoints_xy = keypoints2d.xy[0].cpu().numpy()
            confidence = keypoints2d.conf[0].cpu().numpy()
            valid_mask = confidence > self.confidence_threshold

            if valid_mask.sum() < 4:
                continue  # Not enough points to solve PnP robustly

            image_points = keypoints_xy[valid_mask]

            for j, pose in enumerate(poses):
                keypoints_3d = pose.keypoints  # shape: (num_joints, 3)
                object_points = keypoints_3d[valid_mask]

                image_size = (640, 480)  # Width x Height
                focal_length = image_size[0]

                camera_matrix = np.array([
                    [focal_length, 0, image_size[0] / 2],
                    [0, focal_length, image_size[1] / 2],
                    [0, 0, 1]
                ], dtype=np.float64)

                dist_coeffs = np.zeros((4, 1))

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if not success:
                    continue  # Skip this pair if PnP fails

                projected_points, _ = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs
                )

                projected_points = projected_points.reshape(-1, 2)
                errors = np.linalg.norm(projected_points - image_points, axis=1)
                m_error[i, j] = np.mean(errors)
        row_ind, col_ind = linear_sum_assignment(m_error)
        return m_error[row_ind, col_ind].sum()

    
    def average_similarity(self):
        similarities: List[float] = []
        for keypoints, pose in zip(self.ground_truth, self.poses):
            similarities.append(self.pose_keypoint_similarity(keypoints, pose))
        return np.mean(similarities)
        