import torch
from estimation.util import Estimation
from estimation.preprocess.preprocess_estimation import PreprocessEstimation
from estimation.pose2D.pose_estimation_2D import TwoDPoseEstimation
from estimation.pose3D.pose_estimation_3D import ThreeDPoseEstimation
from utils.process_manager import ProcessManager, StageName
from utils.visualizer import PoseVisualizer
from typing import Optional, Tuple, List


class EstimationPipe:
    def __init__(
        self,
        preprocessor: PreprocessEstimation,
        flatpose: TwoDPoseEstimation,
        poselifting: ThreeDPoseEstimation,
        data_loader: ProcessManager,
    ) -> None:
        self.pipe_classes: List[Tuple[StageName, Estimation]] = [
            ("preprocessor", preprocessor),
            ("flatpose", flatpose),
            ("poselifting", poselifting),
        ]
        self.process_manager: ProcessManager = data_loader
        self.processed_batches: int = 0

    def forward(self) -> torch.Tensor:
        # Check if we should use batch processing for video
        if hasattr(self.process_manager, "video_path") and hasattr(
            self.process_manager, "total_frames"
        ):
            return self.forward_batched()

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

    def forward_batched(self) -> torch.Tensor:
        """Process video in batches to avoid memory issues."""
        print(
            f"Starting batch processing of {self.process_manager.total_frames} frames"
        )

        # Initialize accumulated results for each stage
        stage_accumulator = {}
        batch_idx = 0

        while self.process_manager.processed_frames < self.process_manager.total_frames:
            batch_frames = self.process_manager.get_next_batch()
            if batch_frames is None:
                break

            current_batch_size = batch_frames.shape[0]
            print(
                f"Processing batch {batch_idx + 1}, frames {self.process_manager.processed_frames}-{self.process_manager.processed_frames + current_batch_size}"
            )

            # Process batch through pipeline stages
            current_data = batch_frames
            batch_results = {}

            for stage_name, module in self.pipe_classes:
                if self.process_manager.should_skip_stage(stage_name):
                    continue

                current_data = module.forward(current_data)
                batch_results[stage_name] = current_data

                # Only accumulate results for final stages (skip preprocessor to save memory)
                if stage_name != "preprocessor":
                    if stage_name not in stage_accumulator:
                        stage_accumulator[stage_name] = []
                    stage_accumulator[stage_name].append(current_data.detach().cpu())

            # Update processed frames counter
            self.process_manager.processed_frames += current_batch_size
            batch_idx += 1

            # Memory cleanup
            del current_data
            del batch_results
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Concatenate all accumulated results
        final_results = {}
        for stage_name, batch_list in stage_accumulator.items():
            if batch_list:
                # Ensure all batches have the same number of people (P) by padding if needed
                max_people = max(batch.shape[0] for batch in batch_list)
                padded_batches = []

                print(
                    f"  Stage {stage_name}: max_people={max_people}, num_batches={len(batch_list)}"
                )

                for i, batch in enumerate(batch_list):
                    if batch.shape[0] < max_people:
                        # Pad with zeros to match max_people
                        padding_shape = (max_people - batch.shape[0], *batch.shape[1:])
                        padding = torch.zeros(
                            padding_shape, dtype=batch.dtype, device=batch.device
                        )
                        padded_batch = torch.cat([batch, padding], dim=0)
                        padded_batches.append(padded_batch)
                        print(
                            f"    Batch {i}: padded from {batch.shape} to {padded_batch.shape}"
                        )
                    else:
                        padded_batches.append(batch)

                # Concatenate along the time dimension (axis=1 for shape (P, T, Nk, D))
                concatenated = torch.cat(padded_batches, dim=1)
                final_results[stage_name] = concatenated

                # Store in process manager
                # Find the module for this stage
                stage_module = None
                for s_name, s_module in self.pipe_classes:
                    if s_name == stage_name:
                        stage_module = s_module
                        break

                stage_config = {
                    "stage_name": stage_name,
                    **getattr(stage_module, "config", {}),
                }
                self.process_manager.handle(concatenated, stage_config)

        # Get the final output (last stage)
        final_stage_name = self.pipe_classes[-1][0]
        final_output = final_results[final_stage_name]

        # Final output validation
        assert isinstance(final_output, torch.Tensor)
        if len(final_output.shape) == 4:
            assert final_output.size(2) == 17
            assert final_output.size(3) == 3

        self.processed_batches = batch_idx
        print(
            f"Batch processing complete. Processed {batch_idx} batches, {self.process_manager.processed_frames} frames total."
        )

        return final_output
