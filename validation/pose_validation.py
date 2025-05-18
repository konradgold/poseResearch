import os
from typing import List
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

confidence_threshold = 0.4

class PoseValidation:
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

    @staticmethod
    def pose_keypoint_similarity(keypoints, poses):
        m_error = np.zeros((len(keypoints), poses.shape[0]))
        for i,keypoints2d in enumerate(keypoints):
            keypoints_xy = keypoints2d.xy[0].cpu().numpy()
            confidence = keypoints2d.conf[0].cpu().numpy()
            valid_mask = confidence > confidence_threshold

            if valid_mask.sum() < 4:
                continue  # Not enough points to solve PnP robustly

            image_points = keypoints_xy[valid_mask]
            image_points = np.array(image_points, dtype=np.float32).reshape(-1, 2)

            for j, pose in enumerate(poses):
                keypoints_3d = pose  # shape: (num_joints, 3)
                object_points = keypoints_3d[valid_mask]
                object_points = np.array(object_points, dtype=np.float32).reshape(-1, 3)
                assert object_points.shape[0] == image_points.shape[0]

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
                    dist_coeffs
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
                

                # Compute cosine similarity for each pair
                cos_sim = np.nan_to_num(cosine_similarity(projected_points, image_points))  # shape: (N,)
                #error = np.linalg.norm(projected_points - image_points, axis=1)

                m_error[i, j] -= np.mean(cos_sim.diagonal())
        row_ind, col_ind = linear_sum_assignment(m_error)
        return -m_error[row_ind, col_ind].mean(), -m_error
    
    def preprocess_poses(self, keypoints, pose):
        return keypoints, pose
    
    def average_similarity(self):
        similarities: List[float] = []
        for keypoints, pose in zip(self.ground_truth, self.poses):
            keypoints2d, pose3d = self.preprocess_poses(keypoints, pose)
            similarities.append(self.pose_keypoint_similarity(keypoints2d, pose3d)[0])
        return np.mean(similarities)
        