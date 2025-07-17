from .pose_estimation_2D import TwoDPoseEstimation
import torch
import numpy as np

# Note: Detectron2 must be installed for these imports to work
# Install with: pip install detectron2
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2 import model_zoo
from detectron2.detectron2.engine import DefaultPredictor


class Detectron2Estimation(TwoDPoseEstimation):
    """
    Disclaimer: This class is wip, do not use!
    Detectron2 estimation for 2D poses.
    Available models:
    https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
    """

    @property
    def config(self):
        return {
            "model_config": "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
            "score_threshold": 0.7,
        }

    @property
    def identifier(self):
        return "Detectron2Estimation"

    def __init__(self, model_config: str | None = None, score_threshold: float = 0.7):
        super().__init__()
        self.model_config = model_config or self.config["model_config"]
        self.score_threshold = score_threshold

        # Initialize Detectron2 predictor
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model_config))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_config)
        self.predictor = DefaultPredictor(cfg)

    def _forward(self, images: torch.Tensor) -> torch.Tensor | None:
        """
        Args:
            images (torch.Tensor): Input images of shape (T, H, W, C)
        Returns:
            torch.Tensor: 2D poses of shape (P, T, Nk, D)
        """
        # Ensure images are on CPU and convert to numpy
        if images.is_cuda:
            images = images.cpu()

        # Convert to numpy and ensure correct format (BGR for Detectron2)
        images_np = images.numpy()
        T = images_np.shape[0]

        # Store keypoints for each frame
        all_keypoints = []

        for t in range(T):
            # Detectron2 expects BGR format
            frame = images_np[t]  # Shape: (H, W, C)

            # Run inference
            outputs = self.predictor(frame)["instances"].to("cpu")

            # Extract keypoints
            if outputs.has("pred_keypoints"):
                kps = (
                    outputs.pred_keypoints.numpy()
                )  # Shape: (num_persons, num_keypoints, 3)
                # kps format: (x, y, confidence)
                all_keypoints.append(kps)
            else:
                # No keypoints detected
                all_keypoints.append(
                    np.zeros((0, 17, 3))
                )  # 17 keypoints for COCO format

        # Find maximum number of persons across all frames
        max_persons = (
            max([kps.shape[0] for kps in all_keypoints]) if all_keypoints else 0
        )

        if max_persons == 0:
            # No persons detected in any frame
            return torch.zeros((0, T, 17, 3), dtype=images.dtype)

        # Get number of keypoints (should be 17 for COCO format)
        num_keypoints = (
            all_keypoints[0].shape[1] if all_keypoints[0].shape[0] > 0 else 17
        )

        # Initialize output tensor
        output = torch.zeros((max_persons, T, num_keypoints, 3), dtype=images.dtype)

        # Fill output tensor
        for t, kps in enumerate(all_keypoints):
            num_persons = kps.shape[0]
            if num_persons > 0:
                output[:num_persons, t, :, :] = torch.from_numpy(kps)

        return output
