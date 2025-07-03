import torch
import numpy as np
import cv2
from typing import Any, List, Tuple, Union
from MotionBERT.lib.model.loss import nn
from MotionBERT.lib.utils.learning import load_backbone
from MotionBERT.lib.utils.tools import get_config
from ultralytics import YOLO
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        default="MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        type=str,
        metavar="FILENAME",
        help="checkpoint to evaluate (file name)",
    )
    parser.add_argument(
        "-j", "--json_path", type=str, help="alphapose detection result json path"
    )
    parser.add_argument("-v", "--vid_path", type=str, help="video path")
    parser.add_argument("-o", "--out_path", type=str, help="output path")
    parser.add_argument(
        "--pixel", action="store_true", help="align with pixle coordinates"
    )
    parser.add_argument("--focus", type=int, default=None, help="target person id")
    parser.add_argument(
        "--clip_len", type=int, default=243, help="clip length for network input"
    )
    opts = parser.parse_args()
    return opts


class MotionBertMMPose3DGenerator:
    """
    3D pose generation pipeline combining MMPose for 2D pose detection
    and MotionBERT for 3D pose lifting.
    """

    def __init__(
        self, opts, device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the 3D pose generator.

        Args:
            mmpose_config: MMPose model configuration
            mmpose_checkpoint: Path to MMPose checkpoint (optional)
            motionbert_checkpoint: Path to MotionBERT checkpoint
            device: Device to run inference on
        """
        self.device = device

        # Initialize MMPose

        # Initialize YOLOv11 pose model (assuming ultralytics/yolov11 supports pose)
        self.yolo = YOLO("yolo11n-pose.pt")  # Use the pose model weights for YOLOv11

        # Initialize MotionBERT
        self.motionbert_model = None
        self.load_motionbert(opts)

    def load_motionbert(self, opts):
        """Load MotionBERT model from checkpoint."""
        # Assuming MotionBERT model structure - adjust as needed
        args = get_config(opts.config)
        model_backbone = load_backbone(args)
        if torch.cuda.is_available():
            model_backbone = nn.DataParallel(model_backbone)
            model_backbone = model_backbone.cuda()

        print("Loading checkpoint", opts.evaluate)
        checkpoint = torch.load(
            opts.evaluate, map_location=lambda storage, loc: storage
        )

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint[
            "model_pos"
        ].items():  # or checkpoint if no 'state_dict' key
            new_key = k.replace("module.", "")  # remove 'module.' prefix
            new_state_dict[new_key] = v

        model_backbone.load_state_dict(new_state_dict, strict=True)
        self.motionbert_model = model_backbone
        self.motionbert_model.eval()

    def extract_2d_poses(self, images: Any) -> np.ndarray:
        """
        Extract 2D poses from images using YOLO.

        Args:
            images: Input images (iterable)

        Returns:
            2D keypoints array of shape (T, P, 17, 3) where T is number of frames, P the number of persons maximal encountered.
        """

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = [images]

        # Use YOLO's streaming capability for efficient batch processing
        results = self.yolo(images, stream=True, show=False, verbose=False)

        all_frame_poses = []
        max_persons = 0

        # First pass: collect all poses and find max number of persons
        for result in results:
            frame_poses = []
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                # Extract keypoints for all detected persons
                keypoints = (
                    result.keypoints.data.cpu().numpy()
                )  # Shape: (num_persons, 17, 3)

                for person_keypoints in keypoints:
                    if person_keypoints.shape[0] >= 17:
                        frame_poses.append(
                            person_keypoints[:17]
                        )  # Take first 17 keypoints
                    else:
                        # Pad with zeros if insufficient keypoints
                        padded = np.zeros((17, 3))
                        padded[: person_keypoints.shape[0]] = person_keypoints
                        frame_poses.append(padded)

                max_persons = max(max_persons, len(frame_poses))

            all_frame_poses.append(frame_poses)

        # Ensure max_persons is at least 1
        if max_persons == 0:
            max_persons = 1

        # Second pass: pad all frames to have the same number of persons
        poses_2d = []
        for frame_poses in all_frame_poses:
            if len(frame_poses) == 0:
                # No detections in this frame, add zero poses
                frame_poses = [np.zeros((17, 3)) for _ in range(max_persons)]
            else:
                # Pad with zero poses to reach max_persons
                while len(frame_poses) < max_persons:
                    frame_poses.append(np.zeros((17, 3)))
                # Trim if we have more than max_persons (shouldn't happen)
                frame_poses = frame_poses[:max_persons]

            poses_2d.append(np.array(frame_poses))

        return np.array(poses_2d)  # Shape: (T, P, 17, 3)

    def lift_to_3d(self, poses_2d: np.ndarray) -> np.ndarray:
        """
        Lift 2D poses to 3D using MotionBERT.

        Args:
            poses_2d: 2D poses of shape (T, P, 17, 3)

        Returns:
            3D poses of shape (T, P, 17, 3)
        """
        if self.motionbert_model is None:
            raise RuntimeError("MotionBERT model not loaded")

        # Preprocess 2D poses for MotionBERT
        poses_2d_processed = self.preprocess_2d_poses(poses_2d)
        print(poses_2d_processed.shape)

        with torch.no_grad():
            # Convert to tensor
            poses_tensor = torch.FloatTensor(poses_2d_processed).to(self.device)
            maxlen = (
                int(self.motionbert_model.maxlen)
                if hasattr(self.motionbert_model, "maxlen")
                else poses_tensor.size(0)
            )
            outputs = []
            for start in range(0, poses_tensor.size(0), maxlen):
                end = start + maxlen
                batch_tensor = poses_tensor[start:end].unsqueeze(
                    0
                )  # Add batch dimension
                poses_3d_batch = self.motionbert_model(batch_tensor)
                outputs.append(poses_3d_batch.squeeze(0).cpu())
            poses_3d = torch.cat(outputs, dim=0).numpy()
        return poses_3d

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

        # Normalize each person independently
        for p in range(P):
            person_poses = poses_normalized[:, p, :, :]  # Shape: (T, 17, 3)

            # Extract valid coordinates for this person (where confidence > 0)
            valid_mask = person_poses[..., 2] > 0
            valid_coords = person_poses[valid_mask][:, :2]

            if len(valid_coords) > 0:
                # Center and scale poses for this person
                x_coords = valid_coords[:, 0]
                y_coords = valid_coords[:, 1]

                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()

                # Scale to [-1, 1]
                scale = max(x_max - x_min, y_max - y_min)
                if scale > 0:
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2

                    poses_normalized[:, p, :, 0] = (
                        (poses_normalized[:, p, :, 0] - center_x) / scale * 2
                    )
                    poses_normalized[:, p, :, 1] = (
                        (poses_normalized[:, p, :, 1] - center_y) / scale * 2
                    )

        return poses_normalized

    def generate_3d_poses(
        self,
        input_data: Union[str, List[np.ndarray], np.ndarray],
        use_temporal_smoothing: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D poses from input images or video.

        Args:
            input_data: Video path, list of images, or image array
            use_temporal_smoothing: Whether to apply temporal smoothing

        Returns:
            Tuple of (2D poses, 3D poses)
        """
        # Handle different input types
        if isinstance(input_data, str):
            # Video file path
            images = self._load_video_frames(input_data)
        elif isinstance(input_data, (list, np.ndarray)):
            images = input_data
        else:
            raise ValueError("Unsupported input data type")

        # Extract 2D poses
        poses_2d = self.extract_2d_poses(images)
        print(poses_2d.shape)
        # Apply temporal smoothing if requested
        if use_temporal_smoothing:
            poses_2d = self._apply_temporal_smoothing(poses_2d)

        # Lift to 3D
        poses_3d = self.lift_to_3d(poses_2d)

        return poses_2d, poses_3d

    def _load_video_frames(
        self, video_path: str, max_frames: int = 300
    ) -> List[np.ndarray]:
        """Load frames from video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        frame_count = 0
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1

        cap.release()
        return frames

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

    def save_poses(self, poses_2d: np.ndarray, poses_3d: np.ndarray, output_path: str):
        """Save poses to file."""
        np.savez(output_path, poses_2d=poses_2d, poses_3d=poses_3d)
        print(f"Poses saved to {output_path}")


def main():
    """Example usage of the MotionBertMMPose3DGenerator."""

    # Initialize generator
    opts = parse_args()
    generator = MotionBertMMPose3DGenerator(
        opts=opts,
        # Provide path to MotionBERT checkpoint
    )
    print("3D pose generator initialized successfully!")
    # Example with video file
    poses_2d, poses_3d = generator.generate_3d_poses("/Volumes/KG1TB/data/Untitled.mov")

    print(poses_3d.shape)

    # Example with image array
    # dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # poses_2d, poses_3d = generator.generate_3d_poses([dummy_image])

    # Save results
    # generator.save_poses(poses_2d, poses_3d, "output_poses.npz")

    print("Forward pass was successful!")


if __name__ == "__main__":
    main()
