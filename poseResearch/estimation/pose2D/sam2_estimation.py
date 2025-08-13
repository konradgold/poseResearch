from .pose_estimation_2D import TwoDPoseEstimation
import torch


class SAM2Estimation(TwoDPoseEstimation):
    """
    SAM2 estimation for 2D poses.
    Download checkpoints as described in sam2/checkpoints/download_ckpts.sh.
    Place them in sam2/checkpoints/
    """

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "SAM2Estimation"

    def __init__(self, model: str):
        super().__init__()

    def _forward(self, images: torch.Tensor) -> torch.Tensor | None:
        """
        Args:
            images (torch.Tensor): Input images of shape (T, H, W, C)
        Returns:
            torch.Tensor: 2D poses of shape (P, T, Nk, D) in COCO format
        """
        # TODO: Implement SAM2 estimation
        return images

    def _post_process(self, output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output (torch.Tensor): Output from YOLO in COCO format (P, T, 17, 3)

        Returns:
            torch.Tensor: Post-processed output in h36m format (P, T, 17, 3)
        """
        return output
