from abc import abstractmethod
import torch
from poseResearch.estimation.util import Estimation

class 2DPoseEstimation(Estimation):
    """
    Abstract base class for 2D pose estimation.
    Input: images as a tensor of shape (T, H, W, C)
    Output: 2D poses as a tensor of shape (P, T, Nk, D)
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Input images of shape (T, H, W, C)
        Returns:
            torch.Tensor: 2D poses of shape (P, T, Nk, D)
        """
        pass

    def output_check(self, output) -> bool:
        """
        Checks if the output tensor has the correct shape: (P, T, Nk, D)
        Returns True if valid, False otherwise.
        """
        if not isinstance(output, torch.Tensor):
            return False
        if output.ndim != 4:
            return False
        P, T, Nk, D = output.shape
        expected_Nk = getattr(self, 'num_keypoints', 17)
        expected_D = getattr(self, 'num_dims', 3)
        if D != expected_D or Nk != expected_Nk:
            return False
        if P < 1 or T < 1:
            return False
        return True
