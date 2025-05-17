from ultralytics import YOLO
from typing import Optional
import os
import yaml


class PlanePoseEstimator:
    def __init__(self, model: Optional[str] = None, config: Optional[str] = None):
        if model is not None:
            self.model = YOLO(model)
        else:
            self.model = YOLO("yolo11n-pose.pt")
        
        self.config_dict = dict()
        if config is not None:
            if os.path.isfile(config) and config.lower().endswith(('.yaml', '.yml')):
                with open(config, 'r') as f:
                    self.config_dict = yaml.safe_load(f)
        else:
            self.config_dict["sample_rate"] = 4
            self.config_dict["batch_size"] = None

    def get_pose(self, video_stream):
        i = 0
        images = []
        try:
            while video_stream.isOpened():
                ret, img = video_stream.read()
                if not ret:
                    break
                if i % self.config_dict["sample_rate"] == 0:
                    images.append(img)
        finally:
            video_stream.release()
        
        results = self.model(images)
        return [r.keypoints for r in results]



