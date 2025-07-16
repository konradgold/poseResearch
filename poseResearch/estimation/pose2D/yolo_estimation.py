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

    Note: YOLO outputs poses in COCO format (17 keypoints), which are automatically
    converted to h36m format (17 keypoints) by the post-processing method.
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

    def _post_process(self, output: torch.Tensor) -> torch.Tensor:
        """
        Convert YOLO's COCO format output to h36m format and flip y-axis.

        Args:
            output (torch.Tensor): Output from YOLO in COCO format (P, T, 17, 3)

        Returns:
            torch.Tensor: Post-processed output in h36m format (P, T, 17, 3)
        """
        # Convert from COCO to h36m format using MotionBERT's exact approach

        P, T, Nk_coco, D = output.shape

        # Initialize h36m poses with same shape (17 keypoints)
        poses_h36m = torch.zeros(
            (P, T, 17, D), dtype=output.dtype, device=output.device
        )

        # Apply COCO to h36m conversion for each person and frame
        # Using the exact mapping from MotionBERT's coco2h36m function
        for p in range(P):
            for t in range(T):
                x = output[p, t, :, :]  # (17, 3)

                # COCO to h36m mapping - exact copy of MotionBERT implementation
                # COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
                # H36M: 0-root, 1-rhip, 2-rkne, 3-rank, 4-lhip, 5-lkne, 6-lank, 7-belly, 8-neck, 9-nose, 10-head, 11-lsho, 12-lelb, 13-lwri, 14-rsho, 15-relb, 16-rwri

                poses_h36m[p, t, 0, :] = (
                    x[11, :] + x[12, :]
                ) * 0.5  # root = (lhip + rhip) / 2
                poses_h36m[p, t, 1, :] = x[12, :]  # rhip
                poses_h36m[p, t, 2, :] = x[14, :]  # rkne
                poses_h36m[p, t, 3, :] = x[16, :]  # rank
                poses_h36m[p, t, 4, :] = x[11, :]  # lhip
                poses_h36m[p, t, 5, :] = x[13, :]  # lkne
                poses_h36m[p, t, 6, :] = x[15, :]  # lank
                poses_h36m[p, t, 8, :] = (
                    x[5, :] + x[6, :]
                ) * 0.5  # neck = (lsho + rsho) / 2
                poses_h36m[p, t, 7, :] = (
                    poses_h36m[p, t, 0, :] + poses_h36m[p, t, 8, :]
                ) * 0.5  # belly = (root + neck) / 2
                poses_h36m[p, t, 9, :] = x[0, :]  # nose
                poses_h36m[p, t, 10, :] = (
                    x[1, :] + x[2, :]
                ) * 0.5  # head = (leye + reye) / 2
                poses_h36m[p, t, 11, :] = x[5, :]  # lsho
                poses_h36m[p, t, 12, :] = x[7, :]  # lelb
                poses_h36m[p, t, 13, :] = x[9, :]  # lwri
                poses_h36m[p, t, 14, :] = x[6, :]  # rsho
                poses_h36m[p, t, 15, :] = x[8, :]  # relb
                poses_h36m[p, t, 16, :] = x[10, :]  # rwri

        # Flip y-axis to turn the pose right-side up
        # Assuming the format is (x, y, confidence) where y is at index 1
        poses_h36m[:, :, :, 1] = -poses_h36m[:, :, :, 1]

        return poses_h36m
