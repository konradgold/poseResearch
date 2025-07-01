import torch
from estimation.util import Estimation
from estimation.preprocess.preprocess_estimation import PreprocessEstimation
from estimation.pose2D.pose_estimation_2D import TwoDPoseEstimation
from estimation.pose3D.pose_estimation_3D import ThreeDPoseEstimation
from utils.data_loader import DataLoader
from typing import Tuple, List
from utils.data_loader import StageName


class EstimationPipe:
    def __init__(
        self,
        preprocessor: PreprocessEstimation,
        flatpose: TwoDPoseEstimation,
        poselifting: ThreeDPoseEstimation,
        data_loader: DataLoader,
    ) -> None:
        self.pipe_classes: List[Tuple[StageName, Estimation]] = [
            ("preprocessor", preprocessor),
            ("flatpose", flatpose),
            ("poselifting", poselifting),
        ]
        self.data_loader: DataLoader = data_loader
        self.processed_batches: int = 0

    def forward(self) -> torch.Tensor:
        # Get input data from dataloader
        current_data = self.data_loader.get_current_input()
        if current_data is None:
            raise ValueError(
                "No input data available. Use data_loader.set_input() or load data."
            )

        batch_info = {"original_batch_size": current_data.size(0)}

        # Process through stages using dataloader logic
        for stage_name, module in self.pipe_classes:
            if self.data_loader.should_skip_stage(stage_name):
                continue

            current_data = module.forward(current_data)

            # Store intermediate results in dataloader
            stage_config = {
                "stage_name": stage_name,
                **getattr(module, "config", {}),
            }
            self.data_loader.handle(current_data, stage_config)

        # Final output validation
        assert isinstance(current_data, torch.Tensor)
        if len(current_data.shape) == 4:
            assert current_data.size(2) == 17
            assert current_data.size(3) == 3

        self.processed_batches += 1
        return current_data
