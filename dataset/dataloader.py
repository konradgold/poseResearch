from enum import StrEnum
import cv2
import torch

class CameraAngle(StrEnum):
    top_view = "top_view"
    horizontal_1 = "horizontal_1"
    horizontal_2 = "horizontal_2"
    horizontal_3 = "horizontal_3"

class VideoHandler(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.path = path
        self.video = cv2.VideoCapture(path)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.currentframe = 0
        self.batch_size = 1

    def get_metadata(self):
        return {
            "path": self.path,
            "length": self.length,
            "fps": self.video.get(cv2.CAP_PROP_FPS),
            "width": self.video.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self.video.get(cv2.CAP_PROP_FRAME_HEIGHT),
        }

    def __len__(self):
        return self.length // self.batch_size
    
    def __getitem__(self, idx):
        if self.batch_size > 1:
            return self._get_batch(idx)
        if idx >= self.length:
            raise IndexError("Index out of bounds for video length.")
        
        self.video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.video.read()
        if not ret:
            raise RuntimeError("Failed to read frame from video.")
        
        return frame
    

    def set_batch_size(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        self.batch_size = batch_size

    def _get_batch(self, idx):
        start_frame = idx * self.batch_size
        end_frame = start_frame + self.batch_size
        
        if end_frame > self.length:
            raise IndexError("Batch exceeds video length.")
        
        frames = []
        for frame_idx in range(start_frame, end_frame):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_idx} from video.")
            frames.append(frame)
        return frames

    


