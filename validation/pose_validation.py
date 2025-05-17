import os
from typing import List
from unicodedata import numeric

from numpy import mean as np_mean
from traitlets import Float


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

    def pose_keypoint_similarity(self, keypoints, pose) -> float:
        # TODO
        return 0.
    
    def average_similarity(self):
        similarities: List[float] = []
        for keypoints, pose in zip(self.ground_truth, self.poses):
            similarities.append(self.pose_keypoint_similarity(keypoints, pose))
        return np_mean(similarities)
        