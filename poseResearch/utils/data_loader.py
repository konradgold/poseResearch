import json
import numpy as np
import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path


class DataLoader:
    """
    Minimal dataloader for pipeline stages.
    Stores intermediate results, manages stage flow, and provides input data.
    """

    def __init__(self, save_path: Optional[str] = None):
        self.data_store: Dict[str, Any] = {}
        self.save_path = Path(save_path) if save_path else None
        self.input_data: Optional[torch.Tensor] = None

    def set_input(self, input_data: torch.Tensor) -> None:
        """Set the initial input data (e.g., raw video frames)."""
        self.input_data = input_data

    def get_current_input(self) -> Optional[torch.Tensor]:
        """Get the appropriate input data for the next stage to run."""
        next_stage = self.get_next_stage()

        if next_stage == "preprocessor":
            return self.input_data
        else:
            return self.get_input_for_stage(next_stage)

    def handle(
        self, output: Union[torch.Tensor, np.ndarray], config: Dict[str, Any]
    ) -> None:
        """Store output from a pipeline stage."""
        stage_name = config.get("stage_name", "unknown")

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
            self.save_json(stage_path)

    def get_tensor(self, stage_name: str) -> Optional[torch.Tensor]:
        """Get data as PyTorch tensor from a specific stage."""
        if stage_name in self.data_store:
            data = np.array(self.data_store[stage_name]["data"])
            return torch.from_numpy(data)
        return None

    def run_stage(self, estimation_module, input_stage_name: str) -> torch.Tensor:
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

    def get_next_stage(self) -> str:
        """Determine the next stage to run based on available data."""
        if self.has_stage("flatpose"):
            return "poselifting"
        elif self.has_stage("preprocessor"):
            return "flatpose"
        return "preprocessor"

    def get_input_for_stage(self, stage: str) -> Optional[torch.Tensor]:
        """Get the appropriate input data for a given stage."""
        input_stages = {"flatpose": "preprocessor", "poselifting": "flatpose"}
        input_stage = input_stages.get(stage)
        return self.get_tensor(input_stage) if input_stage else None

    def should_skip_stage(self, stage: str) -> bool:
        """Check if a stage should be skipped based on available data."""
        next_stage = self.get_next_stage()
        stage_order = ["preprocessor", "flatpose", "poselifting"]

        if stage not in stage_order or next_stage not in stage_order:
            return False

        return stage_order.index(stage) < stage_order.index(next_stage)

    def has_input_for_next_stage(self) -> bool:
        """Check if we have input data for the next stage to run."""
        next_stage = self.get_next_stage()
        if next_stage == "preprocessor":
            return self.input_data is not None
        else:
            return self.get_input_for_stage(next_stage) is not None

    def save_json(self, filepath: Optional[str] = None) -> None:
        """Save all stored data to JSON."""
        save_path = Path(filepath) if filepath else self.save_path
        if save_path is None:
            raise ValueError("No save path provided")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self.data_store, f, indent=2)

    def load_json(self, filepath: str) -> None:
        """Load data from JSON."""
        with open(filepath, "r") as f:
            self.data_store = json.load(f)

    def has_stage(self, stage_name: str) -> bool:
        """Check if stage data exists."""
        return stage_name in self.data_store
