from abc import abstractmethod
import torch

class PreprocessEstimation(Estimation):
    """
    Abstract base class for preprocessing steps before 2D pose estimation.
    Input: images as a tensor of shape (T, H, W, C)
    Output: images as a tensor of shape (T, H, W, C)
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Input images of shape (T, H, W, C)
        Returns:
            torch.Tensor: Output images of shape (T, H, W, C)
        """
        pass

    def output_check(self, output) -> bool:
        """
        Checks if the output tensor has the correct shape: (T, H, W, C)
        Returns True if valid, False otherwise.
        """
        if not isinstance(output, torch.Tensor):
            return False
        if output.ndim != 4:
            return False
        T, H, W, C = output.shape
        expected_C = getattr(self, 'num_channels', 3)
        if C != expected_C:
            return False
        if T < 1 or H < 1 or W < 1:
            return False
        return True 