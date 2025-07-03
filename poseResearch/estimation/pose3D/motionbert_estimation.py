import imageio
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import from_coco_to_hm36
from MotionBERT.lib.utils.tools import get_config
from MotionBERT.lib.utils.learning import load_backbone
from MotionBERT.lib.data.dataset_wild import WildDetDataset
from MotionBERT.lib.utils.vismo import render_and_save
from .pose_estimation_3D import ThreeDPoseEstimation


class MotionBERTEstimation(ThreeDPoseEstimation):
    """
    Class for MotionBERT 3D pose estimation.
    Input: 2D poses as a tensor of shape (P, T, 17, 3)
    Output: 3D poses as a tensor of shape (P, T, 18, 3)
    """

    def __init__(
        self,
        config_path: str = "MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml",
        checkpoint_path: str = "MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        vid_path: str = "poseResearch/fem1_t1_preview.mp4",
        json_path: str = "poseResearch/dataloader/results_flatpose.json",
        out_path: str = "poseResearch/results",
        pixel: bool = False,
        rootrel: bool = True,
        gt_2d: bool = False,
        temporal_smoothing: bool = False,
    ):
        super().__init__()
        # Get the project root directory (assuming this file is in poseResearch/estimation/pose3D/)
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )

        # Convert relative paths to absolute paths
        self.config_path = (
            os.path.join(project_root, config_path)
            if not os.path.isabs(config_path)
            else config_path
        )
        self.checkpoint_path = (
            os.path.join(project_root, checkpoint_path)
            if not os.path.isabs(checkpoint_path)
            else checkpoint_path
        )
        self.vid_path = (
            os.path.join(project_root, vid_path)
            if not os.path.isabs(vid_path)
            else vid_path
        )
        self.json_path = (
            os.path.join(project_root, json_path)
            if not os.path.isabs(json_path)
            else json_path
        )
        self.out_path = (
            os.path.join(project_root, out_path)
            if not os.path.isabs(out_path)
            else out_path
        )
        self.pixel = pixel
        self.rootrel = rootrel
        self.gt_2d = gt_2d
        self.temporal_smoothing = temporal_smoothing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "MotionBERTEstimation"

    def preprocess_2d_poses(self, poses_2d: np.ndarray) -> np.ndarray:
        """
        Preprocess 2D poses for MotionBERT input.

        Args:
            poses_2d: Raw 2D poses of shape (T, P, 17, 3)

        Returns:
            Preprocessed poses ready for MotionBERT
        """
        T, P = poses_2d.shape[:2]
        poses_normalized = poses_2d.copy()
        # Unfold dimensions 0 (T) and 1 (P) if any of dim 4 (17 keypoints), 0 (x coordinate) are nonzero
        # That is, keep only frames/persons where any keypoint x != 0
        # poses_2d shape: (T, P, 17, 3)
        nonzero_mask = np.any(poses_normalized[..., 0] != 0, axis=2)  # shape (T, P)
        # Find indices where any person in a frame has nonzero keypoints
        valid_frames = np.where(np.any(nonzero_mask, axis=1))[0]
        valid_persons = np.where(np.any(nonzero_mask, axis=0))[0]
        poses_normalized = poses_normalized[valid_frames][:, valid_persons]
        # Reshape to (T*P, 17, 3) for MotionBERT input
        poses_normalized = poses_normalized.reshape(
            -1, poses_normalized.shape[2], poses_normalized.shape[3]
        )
        return poses_normalized

    def _apply_temporal_smoothing(
        self, poses_2d: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        """Apply temporal smoothing to 2D poses."""
        if len(poses_2d) < window_size:
            return poses_2d

        smoothed = poses_2d.copy()
        half_window = window_size // 2

        for i in range(half_window, len(poses_2d) - half_window):
            window_poses = poses_2d[i - half_window : i + half_window + 1]
            # Simple moving average
            smoothed[i] = np.mean(window_poses, axis=0)

        return smoothed

    def infer_wild(
        self,
        poses_2d: np.ndarray,
        config_path: str = "MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml",
        checkpoint_path: str = "MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        vid_path: str = "poseResearch/fem1_t1_preview.mp4",
        json_path: str = "poseResearch/dataloader/results_flatpose.json",
        out_path: str = "poseResearch/results",
        pixel: bool = False,
        rootrel: bool = True,
        gt_2d: bool = False,
    ) -> np.ndarray:
        print("Running MotionBERT inference...")
        args = get_config(config_path)
        model_backbone = load_backbone(args)
        if torch.cuda.is_available():
            model_backbone = nn.DataParallel(model_backbone)
            model_backbone = model_backbone.cuda()
        print("Loading checkpoint", checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint[
            "model_pos"
        ].items():  # or checkpoint if no 'state_dict' key
            new_key = k.replace("module.", "")  # remove 'module.' prefix
            new_state_dict[new_key] = v

        model_backbone.load_state_dict(new_state_dict, strict=True)
        model_pos = model_backbone
        model_pos.eval()

        with torch.no_grad():
            # Convert to tensor
            poses_tensor = torch.FloatTensor(poses_2d).to(self.device)
            maxlen = (
                int(model_pos.maxlen)
                if hasattr(model_pos, "maxlen")
                and isinstance(model_pos.maxlen, (int, float, torch.Tensor))
                else poses_tensor.size(0)
            )
            outputs = []
            for start in range(0, poses_tensor.size(0), maxlen):
                end = start + maxlen
                batch_tensor = poses_tensor[start:end].unsqueeze(
                    0
                )  # Add batch dimension
                poses_3d_batch = model_pos(batch_tensor)
                outputs.append(poses_3d_batch.squeeze(0).cpu())
            poses_3d = torch.cat(outputs, dim=0).numpy()
        return poses_3d

    def _forward(self, poses_2d: torch.Tensor) -> torch.Tensor:
        # convert poses_2d to numpy array
        poses_2d = poses_2d.cpu().permute(1, 0, 2, 3)
        poses_np = poses_2d.numpy()
        if self.temporal_smoothing:
            poses_np = self._apply_temporal_smoothing(poses_np)
        poses_np = self.preprocess_2d_poses(poses_np)
        poses_3d = self.infer_wild(
            poses_2d=poses_np,
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            vid_path=self.vid_path,
            json_path=self.json_path,
            out_path=self.out_path,
            pixel=self.pixel,
            rootrel=self.rootrel,
            gt_2d=self.gt_2d,
        )

        results_tensor = torch.from_numpy(poses_3d)
        return results_tensor
