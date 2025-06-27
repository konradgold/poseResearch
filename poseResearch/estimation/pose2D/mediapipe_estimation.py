from .pose_estimation_2D import TwoDPoseEstimation
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeEstimation(TwoDPoseEstimation):
    """
    WORK IN PROGRESS. DO NOT USE.
    Mediapipe estimation for 2D poses. More info: https://pypi.org/project/mediapipe/
    Available models:
    https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
    Example: `pose_landmarker_lite.task`
    """

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "MediaPipeEstimation"

    def __init__(self, model_path: str):
        super().__init__()
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            output_segmentation_masks=True,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(self.options)

    def _forward(self, images: torch.Tensor) -> torch.Tensor | None:
        """
        Args:
            images (torch.Tensor): Input images of shape (T, H, W, C)
        Returns:
            torch.Tensor: 2D poses of shape (P, T, Nk, D)
        """
        # Ensure images are on CPU
        print(
            f"{self.identifier} is not tested yet because of different Python versions, returning None."
        )
        return None
        if images.is_cuda:
            images = images.cpu()
        output = []
        for image in images:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = self.landmarker.detect(image)
            output.append(detection_result)
        # Convert to tensor
        output = torch.tensor(output)
        return output
