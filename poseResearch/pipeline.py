import torch
from estimation.preprocess.preprocess_estimation import PreprocessEstimation
from estimation.pose2D.pose_estimation_2D import TwoDPoseEstimation
from estimation.pose3D.pose_estimation_3D import ThreeDPoseEstimation
from poseResearch.utils.data_loader import DataLoader
from utils.visualizer import PoseVisualizer
from typing import Optional, Tuple, List, Generator, Dict, Any


class EstimationPipe:
    def __init__(
        self,
        preprocessor: PreprocessEstimation,
        flatpose: TwoDPoseEstimation,
        poselifting: ThreeDPoseEstimation,
        data_loader: DataLoader,
        visualizer_2d: Optional[PoseVisualizer] = None,
        visualizer_3d: Optional[PoseVisualizer] = None,
    ) -> None:
        self.pipe_classes: List[Tuple[str, Any]] = [
            ("preprocessor", preprocessor),
            ("flatpose", flatpose),
            ("poselifting", poselifting),
        ]
        self.data_loader: DataLoader = data_loader
        self.visualizer_2d: Optional[PoseVisualizer] = visualizer_2d
        self.visualizer_3d: Optional[PoseVisualizer] = visualizer_3d
        self.processed_batches: int = 0

    def forward(self, dataloader: Any) -> Generator[torch.Tensor, None, None]:
        for batch_idx, batch in enumerate(dataloader):
            current_data = batch
            batch_info = {"batch_idx": batch_idx, "original_batch_size": batch.size(0)}

            # Use dataloader to determine stage flow
            start_stage = self.data_loader.get_next_stage()
            if start_stage != "preprocessor":
                stored_data = self.data_loader.get_input_for_stage(start_stage)
                if stored_data is not None:
                    current_data = stored_data

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

                # Use separate visualizers for different stages
                if stage_name == "flatpose" and self.visualizer_2d:
                    if self.visualizer_2d.should_visualize(stage_name, batch_idx):
                        self.visualizer_2d.visualize_2d_poses(
                            current_data, batch_info, stage_name
                        )

                elif stage_name == "poselifting" and self.visualizer_3d:
                    if self.visualizer_3d.should_visualize(stage_name, batch_idx):
                        self.visualizer_3d.visualize_3d_poses(
                            current_data, batch_info, stage_name
                        )

            # Final output validation
            assert isinstance(current_data, torch.Tensor)
            # shape (#persons in batch, #frames, 17, 3)
            assert current_data.size(0) == batch.size(0)
            assert current_data.size(2) == 17
            assert current_data.size(3) == 3

            self.processed_batches += 1
            yield current_data
