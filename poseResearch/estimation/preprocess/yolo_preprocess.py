from .preprocess_estimation import PreprocessEstimation
import cv2
import numpy as np
import os
import torch


class YOLOPreprocess(PreprocessEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "YOLOPreprocess"

    def _input_video_to_tensor(self, video_path: str, num_frames: int) -> torch.Tensor:
        """
        Args:
            video_path (str): Path to the video.
            num_frames (int): Number of frames to read from the video.
        Returns:
            torch.Tensor: Input images of shape (T, H, W, C)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        while count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame to target size (width, height)
            frame = cv2.resize(frame, (640, 640))
            # Convert BGR (OpenCV) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to float32 and normalize to [0, 1]
            frame = frame.astype("float32") / 255.0
            frames.append(frame)
            count += 1
        cap.release()
        if len(frames) == 0:
            raise ValueError("No frames read from video.")
        frames_np = np.array(frames)
        frames_tensor = torch.tensor(frames_np)
        return frames_tensor

    def input_video_to_tensor(self, video_path: str, num_frames: int) -> torch.Tensor:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        valid_extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".flv",
            ".wmv",
            ".mpeg",
            ".mpg",
        }
        _, ext = os.path.splitext(video_path)
        if ext.lower() not in valid_extensions:
            raise ValueError(
                f"File {video_path} does not have a valid video extension: {ext}"
            )
        return self._input_video_to_tensor(video_path, num_frames)

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        return images
