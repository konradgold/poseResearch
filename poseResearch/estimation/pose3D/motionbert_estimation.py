import torch
from .pose_estimation_3D import ThreeDPoseEstimation


class MotionBERTEstimation(ThreeDPoseEstimation):
    """
    WORK IN PROGRESS. DO NOT USE.
    Abstract base class for 3D pose estimation.
    Input: 2D poses as a tensor of shape (P, T, Nk, D)
    Output: (to be defined by subclasses)
    """

    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(
            self.checkpoint_path, map_location=lambda storage, loc: storage
        )

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "MotionBERTEstimation"

    def _forward(self, poses_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            poses_2d (torch.Tensor): Input 2D poses of shape (P, T, Nk, D)
        Returns:
            torch.Tensor: Output tensor (shape defined by subclass)
        """
        return poses_2d
