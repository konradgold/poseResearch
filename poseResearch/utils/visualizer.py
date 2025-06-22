from abc import ABC, abstractmethod
import torch


class PoseVisualizer(ABC):
    """Abstract base class for visualizing pose estimation outputs"""
    
    @abstractmethod
    def visualize_2d_poses(self, poses_2d: torch.Tensor, batch_info: dict, stage_name: str):
        """
        Visualize 2D pose estimations (flatpose output)
        
        Args:
            poses_2d: Tensor of shape (batch_size, frames, 17, 2) for 2D keypoints
            batch_info: Dictionary containing batch metadata
            stage_name: Name of the pipeline stage
        """
        pass
    
    @abstractmethod
    def visualize_3d_poses(self, poses_3d: torch.Tensor, batch_info: dict, stage_name: str):
        """
        Visualize 3D pose estimations (poselifting output)
        
        Args:
            poses_3d: Tensor of shape (batch_size, frames, 17, 3) for 3D keypoints
            batch_info: Dictionary containing batch metadata  
            stage_name: Name of the pipeline stage
        """
        pass
    
    @abstractmethod
    def should_visualize(self, stage_name: str, batch_idx: int) -> bool:
        """
        Determine if visualization should be performed for this stage/batch
        
        Args:
            stage_name: Name of the pipeline stage
            batch_idx: Index of the current batch
            
        Returns:
            bool: Whether to perform visualization
        """
        pass 