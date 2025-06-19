from abc import abstractmethod
import torch
from ..util import Estimation


class ThreeDPoseEstimation(Estimation):
    """
    Abstract base class for 3D pose estimation.
    Input: 2D poses as a tensor of shape (P, T, Nk, D)
    Output: (to be defined by subclasses)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _forward(self, poses_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            poses_2d (torch.Tensor): Input 2D poses of shape (P, T, Nk, D)
        Returns:
            torch.Tensor: Output tensor (shape defined by subclass)
        """
        pass

    def output_check(self, input_tensor) -> bool:
        """
        Checks if the input tensor has the correct shape: (P, T, Nk, D)
        Returns True if valid, False otherwise.
        """
        if not isinstance(input_tensor, torch.Tensor):
            return False
        if input_tensor.ndim != 4:
            return False
        P, T, Nk, D = input_tensor.shape
        expected_Nk = getattr(self, "num_keypoints", 17)
        expected_D = getattr(self, "num_dims", 3)
        if D != expected_D or Nk != expected_Nk:
            return False
        if P < 1 or T < 1:
            return False
        return True
