from .pose_estimation_2D import TwoDPoseEstimation
import torch
from alphapose.models import builder


class AlphaPoseEstimation(TwoDPoseEstimation):
    """
    AlphaPose estimation for 2D poses. More info: https://pypi.org/project/alphapipe/
    Available models:
    https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
    Example: `256x192_res50_lr1e-3_1x.yaml`
    """

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "AlphaPoseEstimation"

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.model = builder.build_sppe(self.model_path)

    def _forward(self, images: torch.Tensor) -> torch.Tensor | None:
        """
        Args:
            images (torch.Tensor): Input images of shape (T, H, W, C)
        Returns:
            torch.Tensor: 2D poses of shape (P, T, Nk, D)
        """
        # Ensure images are on CPU
        print(f"{self.identifier} is not tested yet, returning None.")
        return None
        if images.is_cuda:
            images = images.cpu()
        output = []
        for image in images:
            output.append(image)
        # Convert to tensor
        output = torch.tensor(output)
        return output
