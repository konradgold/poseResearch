import os
import torch
from .utils import from_coco_to_hm36
from MotionBERT.infer_wild import infer_wild
from .pose_estimation_3D import ThreeDPoseEstimation


class MotionBERTEstimation(ThreeDPoseEstimation):
    """
    WORK IN PROGRESS. DO NOT USE.
    Abstract base class for 3D pose estimation.
    Input: 2D poses as a tensor of shape (P, T, Nk, D)
    Output: (to be defined by subclasses)
    """

    def __init__(
        self,
        config_path: str = "MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml",
        checkpoint_path: str = "poseResearch/estimation/pose3D/lib/MotionBERT/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        vid_path: str = "fem1_t1_preview.mp4",
        json_path: str = "dataloader/results_2d.json",
        out_path: str = "results",
        pixel: bool = False,
        rootrel: bool = True,
        gt_2d: bool = False,
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

    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "MotionBERTEstimation"

    def _forward(self, poses_2d: torch.Tensor) -> torch.Tensor:
        # execute the infer_wild.py script
        poses_2d = from_coco_to_hm36(poses_2d)
        poses_3d = infer_wild(
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
