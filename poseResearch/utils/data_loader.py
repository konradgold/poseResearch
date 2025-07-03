import cv2
from cv2.typing import MatLike
import json
import numpy as np
import os
import torch
from typing import Dict, Any, Optional, Union, Literal
from pathlib import Path

# Define the allowed stage names as a type
StageName = Literal["input", "preprocessor", "flatpose", "poselifting", "future"]


class DataLoader:
    """
    Minimal dataloader for pipeline stages.
    Stores intermediate results, manages stage flow, and provides input data.
    """

    def __init__(self, save_path: Optional[str] = None):
        self.data_store: Dict[StageName, Any] = {}
        self.save_path = Path(save_path) if save_path else None

    def set_input(self, input_data: torch.Tensor) -> None:
        """Set the initial input data (e.g., raw video frames)."""
        # Store input data in the same store as stage outputs
        if isinstance(input_data, torch.Tensor):
            data = input_data.detach().cpu().numpy()
        else:
            data = np.array(input_data)

        self.data_store["input"] = {
            "data": data.tolist(),
            "shape": list(data.shape),
            "config": {"stage_name": "input", "description": "Raw input data"},
        }

    def video_to_tensor(
        self, video_path: str, num_frames: Optional[int] = None
    ) -> torch.Tensor:
        """Convert a video to a tensor in BCHW format, using batches for memory efficiency."""
        cap = cv2.VideoCapture(video_path)
        print(f"Read video from {video_path}")
        frames: list[MatLike] = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret or num_frames is not None and count >= num_frames:
                break
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
        print(f"Read {frames_tensor.shape[0]} frames")
        return frames_tensor

    def set_input_from_video(
        self, video_path: str | Path, num_frames: Optional[int] = None
    ) -> None:
        """Set the input data from a video file."""
        resolved_video_path = Path(video_path)
        if not resolved_video_path.exists():
            print(f"Video file not found: {video_path}, search in the project root.")
            # Search in the project root (only works if this file is in poseResearch/utils/)
            relative_video_path = Path(__file__).parent.parent / video_path
            if not relative_video_path.exists():
                raise FileNotFoundError(f"Video file not found: {relative_video_path}")
            resolved_video_path = relative_video_path

        # Check whether video exists
        if not resolved_video_path.exists():
            raise FileNotFoundError(f"Video file not found: {resolved_video_path}")

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
        video_frames = self.video_to_tensor(str(resolved_video_path), num_frames)
        self.set_input(video_frames)

    def get_current_input(self) -> Optional[torch.Tensor]:
        """Get the appropriate input data for the next stage to run."""
        next_stage = self.get_next_stage()

        if next_stage == "preprocessor":
            return self.get_tensor("input")
        else:
            return self.get_input_for_stage(next_stage)

    def handle(
        self, output: Union[torch.Tensor, np.ndarray], config: Dict[str, Any]
    ) -> None:
        """Store output from a pipeline stage."""
        stage_name = config.get("stage_name")

        # Validate stage_name
        if stage_name not in ("preprocessor", "flatpose", "poselifting"):
            raise ValueError(
                f"Invalid stage_name: {stage_name}. Must be one of 'preprocessor', 'flatpose', 'poselifting'"
            )

        # Convert tensor to numpy for JSON serialization
        if isinstance(output, torch.Tensor):
            data = output.detach().cpu().numpy()
        else:
            data = np.array(output)

        # Store data with metadata
        self.data_store[stage_name] = {
            "data": data.tolist(),
            "shape": list(data.shape),
            "config": config,
        }

        # Auto-save if save_path is provided
        if self.save_path:
            # Create stage-specific filename in dataloader folder
            stage_filename = f"results_{stage_name}.json"
            stage_path = self.save_path.parent / "dataloader" / stage_filename
            self.save_json(str(stage_path), stage_name)

    def get_tensor(self, stage_name: StageName) -> Optional[torch.Tensor]:
        """Get data as PyTorch tensor from a specific stage."""
        if stage_name in self.data_store:
            data = np.array(self.data_store[stage_name]["data"])
            return torch.from_numpy(data)
        return None

    def run_stage(self, estimation_module, input_stage_name: StageName) -> torch.Tensor:
        """Run a pipeline stage using stored data as input."""
        input_data = self.get_tensor(input_stage_name)
        if input_data is None:
            raise ValueError(f"No data found for stage: {input_stage_name}")

        output = estimation_module.forward(input_data)

        # Store the output
        output_config = {
            "stage_name": estimation_module.identifier,
            "input_from": input_stage_name,
            **estimation_module.config,
        }
        self.handle(output, output_config)

        return output

    def get_next_stage(self) -> StageName:
        """Determine the next stage to run based on available data."""
        if self.has_stage("flatpose"):
            return "poselifting"
        elif self.has_stage("preprocessor"):
            return "flatpose"
        elif self.has_stage("input"):
            return "preprocessor"
        elif self.has_stage("poselifting"):
            return "future"
        return "preprocessor"

    def get_input_for_stage(self, stage: StageName) -> Optional[torch.Tensor]:
        """Get the appropriate input data for a given stage."""
        input_stages: dict[StageName, StageName] = {
            "flatpose": "preprocessor",
            "poselifting": "flatpose",
            "future": "poselifting",
        }
        input_stage: Optional[StageName] = input_stages.get(stage)
        if input_stage is None:
            raise ValueError(f"No input stage found for stage: {stage}")
        return self.get_tensor(input_stage)

    def should_skip_stage(self, stage: StageName) -> bool:
        """Check if a stage should be skipped based on available data."""
        next_stage = self.get_next_stage()
        stage_order = ["preprocessor", "flatpose", "poselifting", "future"]

        if stage not in stage_order or next_stage not in stage_order:
            return False

        return stage_order.index(stage) < stage_order.index(next_stage)

    def has_input_for_next_stage(self) -> bool:
        """Check if we have input data for the next stage to run."""
        next_stage = self.get_next_stage()
        if next_stage == "preprocessor":
            return "input" in self.data_store
        else:
            return self.get_input_for_stage(next_stage) is not None

    def save_json(
        self, filepath: Optional[str] = None, stage_name: Optional[StageName] = None
    ) -> None:
        """Save all stored data to JSON.
        If stage_name is provided, save only the data for that stage.
        """
        if stage_name == "preprocessor":
            return
        save_path = Path(filepath) if filepath else self.save_path
        if save_path is None:
            raise ValueError("No save path provided")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        if stage_name:
            # Wrap the stage data with the stage name as key to maintain consistent structure
            data_to_save = {stage_name: self.data_store[stage_name]}
        else:
            data_to_save = self.data_store

        with open(save_path, "w") as f:
            json.dump(data_to_save, f, indent=2)

    def load_json(self, filepath: str) -> None:
        """Load data from JSON."""
        with open(filepath, "r") as f:
            self.data_store = json.load(f)

    def has_stage(self, stage_name: StageName) -> bool:
        """Check if stage data exists."""
        return stage_name in self.data_store
