import numpy as np
import os
import torch
import torch.nn as nn
from MotionBERT.lib.utils.tools import get_config
from MotionBERT.lib.utils.learning import load_backbone
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

    def _apply_rotation(
        self, poses_3d: torch.Tensor, axis: str = "z", angle_degrees: float = -90
    ) -> torch.Tensor:
        """
        Apply rotation around specified axis to correct orientation.

        Args:
            poses_3d: Input 3D poses of shape (..., 3) where last dimension is [x, y, z]
            axis: Rotation axis ('x', 'y', or 'z')
            angle_degrees: Rotation angle in degrees (negative for clockwise)

        Returns:
            Rotated poses
        """
        import math

        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        # Create rotation matrix based on axis
        if axis.lower() == "x":
            # X-axis rotation matrix
            rotation_matrix = torch.tensor(
                [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]],
                dtype=poses_3d.dtype,
                device=poses_3d.device,
            )
        elif axis.lower() == "y":
            # Y-axis rotation matrix
            rotation_matrix = torch.tensor(
                [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]],
                dtype=poses_3d.dtype,
                device=poses_3d.device,
            )
        elif axis.lower() == "z":
            # Z-axis rotation matrix
            rotation_matrix = torch.tensor(
                [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
                dtype=poses_3d.dtype,
                device=poses_3d.device,
            )
        else:
            raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'.")

        # Apply rotation to the last dimension (x, y, z coordinates)
        # poses_3d shape: (..., 3)
        original_shape = poses_3d.shape
        poses_flat = poses_3d.view(-1, 3)  # Flatten to (N, 3)
        rotated_flat = torch.matmul(poses_flat, rotation_matrix.T)  # Apply rotation
        rotated_poses = rotated_flat.view(original_shape)  # Restore original shape

        return rotated_poses

    def _apply_multiple_rotations(
        self, poses_3d: torch.Tensor, rotations: list
    ) -> torch.Tensor:
        """
        Apply multiple rotations in sequence.

        Args:
            poses_3d: Input 3D poses of shape (..., 3)
            rotations: List of tuples (axis, angle_degrees) e.g., [('z', -90), ('x', 45)]

        Returns:
            Rotated poses
        """
        result = poses_3d
        for axis, angle in rotations:
            result = self._apply_rotation(result, axis=axis, angle_degrees=angle)
        return result

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
        # TODO:
        # Due to flattening poses_2d into 3 dimensions during preprocessing, we only have 3 here as well.
        # Find out whether this is a MotionBERT problem. If not, this has to be fixed or MotionBERT is only usable for a single person.
        if results_tensor.ndim == 3:
            results_tensor = results_tensor.unsqueeze(0)

        # Apply rotation correction to fix the 90-degree z-axis rotation
        # results_tensor = self._apply_multiple_rotations(
        #    results_tensor, rotations=[("z", 0), ("y", 0), ("x", 0)]
        # )

        return results_tensor
