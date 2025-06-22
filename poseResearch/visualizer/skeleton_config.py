from typing import Dict, List, Tuple
from ..utils.skeleton_config import SkeletonConfig


class AnatomicalSkeletonConfig(SkeletonConfig):
    """Anatomical skeleton configuration with 17 keypoints"""
    
    def get_keypoint_names(self) -> List[str]:
        return [
            'root', 'right_hip', 'right_knee', 'right_foot',
            'left_hip', 'left_knee', 'left_foot', 'spine',
            'thorax', 'neck_base', 'head', 'left_shoulder',
            'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist'
        ]
    
    def get_keypoint_id2name(self) -> Dict[int, str]:
        return {
            0: "root", 1: "right_hip", 2: "right_knee", 3: "right_foot",
            4: "left_hip", 5: "left_knee", 6: "left_foot", 7: "spine",
            8: "thorax", 9: "neck_base", 10: "head", 11: "left_shoulder",
            12: "left_elbow", 13: "left_wrist", 14: "right_shoulder", 
            15: "right_elbow", 16: "right_wrist"
        }
    
    def get_keypoint_name2id(self) -> Dict[str, int]:
        return {
            "root": 0, "right_hip": 1, "right_knee": 2, "right_foot": 3,
            "left_hip": 4, "left_knee": 5, "left_foot": 6, "spine": 7,
            "thorax": 8, "neck_base": 9, "head": 10, "left_shoulder": 11,
            "left_elbow": 12, "left_wrist": 13, "right_shoulder": 14,
            "right_elbow": 15, "right_wrist": 16
        }
    
    def get_skeleton_links(self) -> List[Tuple[int, int]]:
        return [
            # Head and neck
            (10, 9),  # head -> neck_base
            # Torso spine
            (9, 8),   # neck_base -> thorax
            (8, 7),   # thorax -> spine
            (7, 0),   # spine -> root
            # Left arm
            (8, 11),  # thorax -> left_shoulder
            (11, 12), # left_shoulder -> left_elbow
            (12, 13), # left_elbow -> left_wrist
            # Right arm
            (8, 14),  # thorax -> right_shoulder
            (14, 15), # right_shoulder -> right_elbow
            (15, 16), # right_elbow -> right_wrist
            # Left leg
            (0, 4),   # root -> left_hip
            (4, 5),   # left_hip -> left_knee
            (5, 6),   # left_knee -> left_foot
            # Right leg
            (0, 1),   # root -> right_hip
            (1, 2),   # right_hip -> right_knee
            (2, 3),   # right_knee -> right_foot
        ]
    
    def get_body_part_colors(self) -> Dict[str, str]:
        return {
            'head': '#FF6B6B',        # Red
            'torso': '#4ECDC4',       # Teal
            'left_arm': '#45B7D1',    # Blue
            'right_arm': '#FFA726',   # Orange
            'left_leg': '#AB47BC',    # Purple
            'right_leg': '#66BB6A',   # Green
        }
    
    def get_keypoint_body_parts(self) -> List[str]:
        return [
            'torso',      # 0: root
            'right_leg',  # 1: right_hip
            'right_leg',  # 2: right_knee
            'right_leg',  # 3: right_foot
            'left_leg',   # 4: left_hip
            'left_leg',   # 5: left_knee
            'left_leg',   # 6: left_foot
            'torso',      # 7: spine
            'torso',      # 8: thorax
            'head',       # 9: neck_base
            'head',       # 10: head
            'left_arm',   # 11: left_shoulder
            'left_arm',   # 12: left_elbow
            'left_arm',   # 13: left_wrist
            'right_arm',  # 14: right_shoulder
            'right_arm',  # 15: right_elbow
            'right_arm',  # 16: right_wrist
        ]


class COCOSkeletonConfig(SkeletonConfig):
    """COCO skeleton configuration with 17 keypoints"""
    
    def get_keypoint_names(self) -> List[str]:
        return [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def get_keypoint_id2name(self) -> Dict[int, str]:
        return {i: name for i, name in enumerate(self.get_keypoint_names())}
    
    def get_keypoint_name2id(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.get_keypoint_names())}
    
    def get_skeleton_links(self) -> List[Tuple[int, int]]:
        return [
            (0, 1), (0, 2), (1, 3), (2, 4),  # head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
            (5, 11), (6, 12), (11, 12),  # torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # legs
        ]
    
    def get_body_part_colors(self) -> Dict[str, str]:
        return {
            'head': '#FF6B6B',        # Red
            'torso': '#4ECDC4',       # Teal
            'left_arm': '#45B7D1',    # Blue
            'right_arm': '#FFA726',   # Orange
            'left_leg': '#AB47BC',    # Purple
            'right_leg': '#66BB6A',   # Green
        }
    
    def get_keypoint_body_parts(self) -> List[str]:
        return [
            'head', 'head', 'head', 'head', 'head',  # 0-4: face
            'torso', 'torso', 'left_arm', 'right_arm',  # 5-8: shoulders, elbows
            'left_arm', 'right_arm', 'torso', 'torso',  # 9-12: wrists, hips
            'left_leg', 'right_leg', 'left_leg', 'right_leg'  # 13-16: knees, ankles
        ]


# Factory function for easy skeleton config creation
def create_skeleton_config(skeleton_type: str) -> SkeletonConfig:
    """Factory function to create skeleton configurations"""
    configs = {
        'anatomical': AnatomicalSkeletonConfig,
        'coco': COCOSkeletonConfig,
    }
    
    if skeleton_type.lower() not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(f"Unknown skeleton type '{skeleton_type}'. Available: {available}")
    
    config = configs[skeleton_type.lower()]()
    config.validate_config()  # Ensure the configuration is valid
    return config 