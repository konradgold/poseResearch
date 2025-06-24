from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np


class SkeletonConfig(ABC):
    """Abstract base class for skeleton configurations"""
    
    @abstractmethod
    def get_keypoint_names(self) -> List[str]:
        """Return ordered list of keypoint names"""
        pass
    
    @abstractmethod
    def get_keypoint_id2name(self) -> Dict[int, str]:
        """Return mapping from keypoint ID to name"""
        pass
    
    @abstractmethod
    def get_keypoint_name2id(self) -> Dict[str, int]:
        """Return mapping from keypoint name to ID"""
        pass
    
    @abstractmethod
    def get_skeleton_links(self) -> List[Tuple[int, int]]:
        """Return list of skeleton connections as (start_id, end_id) tuples"""
        pass
    
    @abstractmethod
    def get_body_part_colors(self) -> Dict[str, str]:
        """Return mapping from body part to color"""
        pass
    
    @abstractmethod
    def get_keypoint_body_parts(self) -> List[str]:
        """Return ordered list of body parts for each keypoint"""
        pass
    
    def get_keypoint_colors(self) -> List[str]:
        """Return colors for each keypoint based on body part mapping"""
        body_part_colors = self.get_body_part_colors()
        keypoint_body_parts = self.get_keypoint_body_parts()
        return [body_part_colors[part] for part in keypoint_body_parts]
    
    def get_skeleton_colors(self) -> List[str]:
        """Return colors for skeleton links based on starting keypoint"""
        keypoint_colors = self.get_keypoint_colors()
        skeleton_links = self.get_skeleton_links()
        return [keypoint_colors[start_idx] for start_idx, _ in skeleton_links]
    
    def get_num_keypoints(self) -> int:
        """Return total number of keypoints"""
        return len(self.get_keypoint_names())
    
    def validate_config(self) -> bool:
        """Validate that the skeleton configuration is consistent"""
        keypoint_names = self.get_keypoint_names()
        keypoint_id2name = self.get_keypoint_id2name()
        keypoint_name2id = self.get_keypoint_name2id()
        keypoint_body_parts = self.get_keypoint_body_parts()
        skeleton_links = self.get_skeleton_links()
        body_part_colors = self.get_body_part_colors()
        
        num_keypoints = len(keypoint_names)
        
        # Check that all mappings have consistent lengths
        if len(keypoint_id2name) != num_keypoints:
            raise ValueError(f"keypoint_id2name has {len(keypoint_id2name)} entries, expected {num_keypoints}")
        
        if len(keypoint_name2id) != num_keypoints:
            raise ValueError(f"keypoint_name2id has {len(keypoint_name2id)} entries, expected {num_keypoints}")
        
        if len(keypoint_body_parts) != num_keypoints:
            raise ValueError(f"keypoint_body_parts has {len(keypoint_body_parts)} entries, expected {num_keypoints}")
        
        # Check that id2name and name2id are consistent
        for i, name in enumerate(keypoint_names):
            if keypoint_id2name.get(i) != name:
                raise ValueError(f"Inconsistent mapping: keypoint_names[{i}] = '{name}', keypoint_id2name[{i}] = '{keypoint_id2name.get(i)}'")
            
            if keypoint_name2id.get(name) != i:
                raise ValueError(f"Inconsistent mapping: '{name}' -> {keypoint_name2id.get(name)}, expected {i}")
        
        # Check that all body parts have colors
        unique_parts = set(keypoint_body_parts)
        for part in unique_parts:
            if part not in body_part_colors:
                raise ValueError(f"Body part '{part}' not found in body_part_colors")
        
        # Check that skeleton links reference valid keypoint indices
        for start_idx, end_idx in skeleton_links:
            if start_idx < 0 or start_idx >= num_keypoints:
                raise ValueError(f"Invalid skeleton link: start_idx {start_idx} out of range [0, {num_keypoints-1}]")
            if end_idx < 0 or end_idx >= num_keypoints:
                raise ValueError(f"Invalid skeleton link: end_idx {end_idx} out of range [0, {num_keypoints-1}]")
        
        return True
    
    def print_info(self):
        """Print detailed information about the skeleton configuration"""
        print(f"=== {self.__class__.__name__} ===")
        print(f"Total keypoints: {self.get_num_keypoints()}")
        print(f"Total skeleton links: {len(self.get_skeleton_links())}")
        
        print("\nKeypoint mapping:")
        keypoint_names = self.get_keypoint_names()
        keypoint_body_parts = self.get_keypoint_body_parts()
        body_part_colors = self.get_body_part_colors()
        
        for i, name in enumerate(keypoint_names):
            part = keypoint_body_parts[i]
            color = body_part_colors[part]
            print(f"  {i:2d}: {name:15} ({part}, {color})")
        
        print("\nSkeleton connections:")
        skeleton_links = self.get_skeleton_links()
        for i, (start, end) in enumerate(skeleton_links):
            start_name = keypoint_names[start]
            end_name = keypoint_names[end]
            print(f"  {i:2d}: {start_name:15} -> {end_name:15}")
        
        print("\nBody part colors:")
        for part, color in self.get_body_part_colors().items():
            print(f"  {part:12}: {color}") 