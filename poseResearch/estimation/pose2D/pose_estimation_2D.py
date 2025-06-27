from abc import abstractmethod
import torch
from ..util import Estimation


class TwoDPoseEstimation(Estimation):
    """
    Abstract base class for 2D pose estimation.
    Input: images as a tensor of shape (T, H, W, C)
    Output: 2D poses as a tensor of shape (P, T, Nk, D) in the COCO format.
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
            print(f"Warning: {self.identifier} returned non-tensor output.")
            return False
        if output.ndim != 4:
            print(
                f"Warning: {self.identifier} returned tensor with {output.ndim} dimensions. Expected 4."
            )
            return False
        P, T, Nk, D = output.shape
        print(f"P: {P}, T: {T}, Nk: {Nk}, D: {D}")
        expected_Nk = getattr(self, "num_keypoints", 17)
        expected_D = getattr(self, "num_dims", 3)
        if D != expected_D or Nk != expected_Nk:
            print(
                f"Warning: {self.identifier} returned tensor with {Nk} keypoints and {D} dimensions. Expected {expected_Nk} keypoints and {expected_D} dimensions."
            )
            if Nk == 0 and D == expected_Nk * expected_D:
                print(
                    f"Warning: {self.identifier} expected output tensor with {expected_Nk} keypoints and {expected_D} dimensions, but got {Nk} keypoints and {D} dimensions. This might occur when no person is detected."
                )
                return True
            return False
        if T < 1:
            print(f"Warning: {self.identifier} returned no frames.")
            return False
        if P < 1:
            print(f"Warning: {self.identifier} returned no persons.")
        return True
