from .pose_estimation_2D import TwoDPoseEstimation
import cv2
import numpy as np
import torch
from mmdeploy_runtime import PoseDetector


class RTMPoseEstimation(TwoDPoseEstimation):
    """
    RTMPose estimation for 2D poses.
    Available models:
    https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose#-model-zoo-
    """

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "RTMPoseEstimation"

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def _forward(self, images: torch.Tensor) -> torch.Tensor | None:
        """
        Args:
            images (torch.Tensor): Input images of shape (T, H, W, C)
        Returns:
            torch.Tensor: 2D poses of shape (P, T, Nk, D)
        """
        print(f"{self.identifier} is not tested yet, returning None.")
        return None
        # Ensure images are on CPU for YOLO
        if images.is_cuda:
            images = images.cpu()

        image_path = "test.jpg"
        img = cv2.imread(image_path)
        detector = PoseDetector(self.model_path)

        # converter (x, y, w, h) -> (left, top, right, bottom)
        args_bbox = np.array([0, 0, 100, 100], dtype=int)
        bbox = np.array(args_bbox, dtype=int)
        bbox[2:] += bbox[:2]
        result = detector(img, bbox)
        print(result)

        _, point_num, _ = result.shape
        points = result[:, :, :2].reshape(point_num, 2)
        for [x, y] in points.astype(int):
            cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

        cv2.imwrite("output_pose.png", img)

        return result
