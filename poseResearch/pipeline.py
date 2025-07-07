import torch
from estimation.util import Estimation
from estimation.preprocess.preprocess_estimation import PreprocessEstimation
from estimation.pose2D.pose_estimation_2D import TwoDPoseEstimation
from estimation.pose3D.pose_estimation_3D import ThreeDPoseEstimation
from poseResearch.utils.process_manager import ProcessManager
from utils.visualizer import PoseVisualizer
from typing import Optional, Tuple, List
from poseResearch.utils.process_manager import StageName


class EstimationPipe:
    def __init__(
        self,
        preprocessor: PreprocessEstimation,
        flatpose: TwoDPoseEstimation,
        poselifting: ThreeDPoseEstimation,
        data_loader: ProcessManager,
        visualizer_2d: Optional[PoseVisualizer] = None,
        visualizer_3d: Optional[PoseVisualizer] = None,
    ) -> None:
        self.pipe_classes: List[Tuple[StageName, Estimation]] = [
            ("preprocessor", preprocessor),
            ("flatpose", flatpose),
            ("poselifting", poselifting),
        ]
        self.process_manager: ProcessManager = data_loader
        self.visualizer_2d: Optional[PoseVisualizer] = visualizer_2d
        self.visualizer_3d: Optional[PoseVisualizer] = visualizer_3d
        self.processed_batches: int = 0

    def forward(self) -> torch.Tensor:
        # Get input data from dataloader
        current_data = self.process_manager.get_current_input()
        if current_data is None:
            raise ValueError(
                "No input data available. Use data_loader.set_input() or load data."
            )

        # Process through stages using dataloader logic
        for stage_name, module in self.pipe_classes:
            if self.process_manager.should_skip_stage(stage_name):
                continue

            current_data = module.forward(current_data)

            # Store intermediate results in dataloader
            stage_config = {
                "stage_name": stage_name,
                **getattr(module, "config", {}),
            }
            self.process_manager.handle(current_data, stage_config)

        # Final output validation
        assert isinstance(current_data, torch.Tensor)
        if len(current_data.shape) == 4:
            assert current_data.size(2) == 17
            assert current_data.size(3) == 3

        self.processed_batches += 1
        return current_data
