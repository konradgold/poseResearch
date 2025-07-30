from .pose_estimation_2D import TwoDPoseEstimation
import torch
from ultralytics import YOLO
from typing import Literal


available__yolo_pose_models = Literal[
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
    "yolo11x-pose.pt",
]


class YOLOEstimation(TwoDPoseEstimation):
    """
    YOLO estimation for 2D poses.
    Available models:
    https://docs.ultralytics.com/de/models/
    Example: `yolo11s-pose.pt`

    Note: YOLO outputs poses in COCO format (17 keypoints). Format conversion to h36m
    is handled by downstream estimation steps (e.g., MotionBERT pre-processing).
    """

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "YOLOEstimation"

    def __init__(self, model: available__yolo_pose_models):
        super().__init__()
        self.model = YOLO(model)

    def _forward(self, images: torch.Tensor) -> torch.Tensor | None:
        """
        Args:
            images (torch.Tensor): Input images of shape (T, H, W, C)
        Returns:
            torch.Tensor: 2D poses of shape (P, T, Nk, D) in COCO format
        """
        # Ensure images are on CPU for YOLO
        if images.is_cuda:
            images = images.cpu()

        # See https://docs.ultralytics.com/modes/predict/#inference-sources for needed shape
        # torch expects (T, C, H, W)
        images_in_shape = images.permute(0, 3, 1, 2)
        T = images_in_shape.shape[0]
        # Run inference
        results = self.model(images_in_shape)
        # Add show=True, save=True to see and save the results

        # Each result corresponds to one image
        # For each image, result.keypoints.data is (num_persons, num_keypoints, 3)
        # We want output shape (P, T, Nk, D) where D=3 (x, y, conf)
        # First, find max number of persons (P) across all frames
        num_persons_per_frame = [r.keypoints.data.shape[0] for r in results]
        P = max(num_persons_per_frame) if num_persons_per_frame else 0
        Nk = results[0].keypoints.data.shape[1] if P > 0 else 17
        D = results[0].keypoints.data.shape[2] if P > 0 else 3
        # Initialize output tensor with zeros
        output = torch.zeros((P, T, Nk, D), dtype=images.dtype)
        for t, r in enumerate(results):
            kpts = r.keypoints.data  # (num_persons, Nk, D)
            num_persons = kpts.shape[0]
            if num_persons > 0:
                output[:num_persons, t, :, :] = kpts

        return output
