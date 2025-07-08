import numpy as np
import os
import torch
from .pose_estimation_3D import ThreeDPoseEstimation


class VideoPose3DEstimation(ThreeDPoseEstimation):
    """
    Class for MotionBERT 3D pose estimation.
    Input: 2D poses as a tensor of shape (P, T, 17, 3)
    Output: 3D poses as a tensor of shape (P, T, 18, 3)
    """

    def __init__(
        self,
        checkpoint_path: str = "MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
    ):
        super().__init__()
        # Get the project root directory (assuming this file is in poseResearch/estimation/pose3D/)
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )

        # Convert relative paths to absolute paths
        self.checkpoint_path = (
            os.path.join(project_root, checkpoint_path)
            if not os.path.isabs(checkpoint_path)
            else checkpoint_path
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "VideoPose3DEstimation"

    def preprocess_2d_poses(self, poses_2d: np.ndarray) -> np.ndarray:
        """
        Preprocess 2D poses for VideoPose3D input.

        Args:
            poses_2d: Raw 2D poses of shape (T, P, 17, 3) in H36M format

        Returns:
            Preprocessed poses ready for VideoPose3D
        """
        return poses_2d

    def _forward(self, poses_2d: torch.Tensor) -> torch.Tensor:
        # convert poses_2d to numpy array
        poses_np = poses_2d.numpy()
        poses_3d = poses_np
        results_tensor = torch.from_numpy(poses_3d)

        return results_tensor
