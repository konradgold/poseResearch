import torch
from estimation.util import Estimation
from utils.output_saver import OutputSaver
from utils.visualizer import PoseVisualizer
from typing import Optional


class EstimationPipe:
    def __init__(
        self,
        preprocessor: Estimation,
        flatpose: Estimation,
        poselifting: Estimation,
        output_saver: OutputSaver,
        visualizer_2d: Optional[PoseVisualizer] = None,
        visualizer_3d: Optional[PoseVisualizer] = None,
    ):
        self.pipe_classes = [
            ("preprocessor", preprocessor),
            ("flatpose", flatpose), 
            ("poselifting", poselifting)
        ]
        self.output_saver = output_saver
        self.visualizer_2d = visualizer_2d
        self.visualizer_3d = visualizer_3d
        self.processed_batches = 0

    def forward(self, dataloader):
        for batch_idx, batch in enumerate(dataloader):
            current_data = batch
            batch_info = {"batch_idx": batch_idx, "original_batch_size": batch.size(0)}
            
            # Process through each stage
            for stage_name, module in self.pipe_classes:
                current_data = module.forward(current_data)
                self.output_saver.handle(current_data, module.config)
                
                # Use separate visualizers for different stages
                if stage_name == "flatpose" and self.visualizer_2d:
                    if self.visualizer_2d.should_visualize(stage_name, batch_idx):
                        self.visualizer_2d.visualize_2d_poses(current_data, batch_info, stage_name)
                        
                elif stage_name == "poselifting" and self.visualizer_3d:
                    if self.visualizer_3d.should_visualize(stage_name, batch_idx):
                        self.visualizer_3d.visualize_3d_poses(current_data, batch_info, stage_name)
            
            # Final output validation
            assert isinstance(current_data, torch.Tensor)
            # shape (#persons in batch, #frames, 17, 3)
            assert current_data.size(0) == batch.size(0)
            assert current_data.size(2) == 17
            assert current_data.size(3) == 3
            
            self.processed_batches += 1
            yield current_data
    
    def create_videos_from_visualizations(self, cleanup_images: bool = True):
        """Create MP4 videos using visualizer's built-in video creation"""
        print("Creating videos from visualization images...")
        all_created_videos = []
        
        # Create videos for 2D visualizations
        if self.visualizer_2d:
            videos_2d = self.visualizer_2d.create_all_videos()
            all_created_videos.extend(videos_2d)
            if cleanup_images and videos_2d:
                self.visualizer_2d.cleanup_images_after_video()
        
        # Create videos for 3D visualizations  
        if self.visualizer_3d:
            videos_3d = self.visualizer_3d.create_all_videos()
            all_created_videos.extend(videos_3d)
            if cleanup_images and videos_3d:
                self.visualizer_3d.cleanup_images_after_video()
        
        return all_created_videos
