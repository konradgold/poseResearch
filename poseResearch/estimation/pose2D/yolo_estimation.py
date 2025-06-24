from .pose_estimation_2D import TwoDPoseEstimation
import torch
from ultralytics import YOLO


class YOLOEstimation(TwoDPoseEstimation):
    """
    YOLO estimation for 2D poses.
    """

    def __init__(self, model_path: str):
        super().__init__()
        self.model = YOLO(model_path)

    def _forward(self, images: torch.Tensor) -> torch.Tensor | None:
        """
        Args:
            images (torch.Tensor): Input images of shape (T, H, W, C)
        Returns:
            torch.Tensor: 2D poses of shape (P, T, Nk, D)
        """
        # Ensure images are on CPU and in numpy format for YOLO
        if images.is_cuda:
            images = images.cpu()
        images_np = images.numpy()
        T = images_np.shape[0]
        # YOLO expects images as list of HWC numpy arrays
        image_list = [images_np[t] for t in range(T)]
        # Run inference
        results = self.model(image_list)
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
                output[:num_persons, t, :, :] = torch.from_numpy(kpts)
        return output
