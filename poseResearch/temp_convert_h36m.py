#!/usr/bin/env python3
"""
Temporary converter to transform H36M JSON format into anatomical format for visualization
"""

import json
import numpy as np
import torch
import sys
import os

def convert_h36m_to_anatomical(input_file: str, output_file: str = None):
    """
    Convert H36M JSON format to anatomical format for visualization
    
    Args:
        input_file: Path to H36M JSON file
        output_file: Path for output file (optional, defaults to input_file + '_anatomical.json')
    """
    
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_anatomical.json"
    
    print(f"Converting {input_file} to anatomical format...")
    
    try:
        # Load H36M data
        with open(input_file, 'r') as f:
            h36m_data = json.load(f)
        
        # Extract metadata
        meta_info = h36m_data["meta_info"]
        instance_info = h36m_data["instance_info"]
        
        print(f"Dataset: {meta_info['dataset_name']}")
        print(f"Keypoints: {meta_info['num_keypoints']}")
        print(f"Frames: {len(instance_info)}")
        
        # Verify keypoint mapping matches anatomical
        expected_keypoints = [
            'root', 'right_hip', 'right_knee', 'right_foot',
            'left_hip', 'left_knee', 'left_foot', 'spine',
            'thorax', 'neck_base', 'head', 'left_shoulder',
            'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist'
        ]
        
        # Check if keypoint names match
        h36m_keypoints = [meta_info["keypoint_id2name"][str(i)] for i in range(17)]
        if h36m_keypoints != expected_keypoints:
            print("Warning: Keypoint names don't match exactly, but proceeding with conversion...")
            print(f"H36M keypoints: {h36m_keypoints}")
            print(f"Expected keypoints: {expected_keypoints}")
        
        # Extract pose data
        num_frames = len(instance_info)
        poses_3d_list = []
        
        for frame_data in instance_info:
            frame_instances = frame_data["instances"]
            num_people = len(frame_instances)
            
            # Limit to 2 people as requested
            num_people = min(num_people, 2)
            
            frame_poses = []
            for person_idx in range(num_people):
                person_data = frame_instances[person_idx]
                keypoints = person_data["keypoints"]
                
                # Convert keypoints to numpy array
                keypoints_array = np.array(keypoints)  # Shape: (17, 3)
                frame_poses.append(keypoints_array.tolist())
            
            # If only 1 person, pad with zeros for second person
            if num_people == 1:
                zero_pose = np.zeros((17, 3)).tolist()
                frame_poses.append(zero_pose)
            
            poses_3d_list.append(frame_poses)
        
        # Convert to final format: (batch_size=2, frames, keypoints=17, dims=3)
        poses_3d_array = np.array(poses_3d_list)  # Shape: (frames, people, 17, 3)
        poses_3d_array = np.transpose(poses_3d_array, (1, 0, 2, 3))  # Shape: (people, frames, 17, 3)
        
        print(f"Final pose array shape: {poses_3d_array.shape}")
        
        # Create anatomical format output
        anatomical_data = {
            "poses_3d": poses_3d_array.tolist(),
            "metadata": {
                "num_people": poses_3d_array.shape[0],
                "num_frames": poses_3d_array.shape[1],
                "num_keypoints": poses_3d_array.shape[2],
                "skeleton_type": "anatomical",
                "units": "normalized_coordinates",
                "original_dataset": meta_info["dataset_name"],
                "keypoint_names": expected_keypoints,
                "conversion_info": {
                    "converted_from": "h36m_format",
                    "original_file": input_file,
                    "keypoint_mapping_verified": h36m_keypoints == expected_keypoints
                }
            },
            "frame_info": [
                {
                    "frame_id": i,
                    "original_frame_id": instance_info[i]["frame_id"] if i < len(instance_info) else None
                }
                for i in range(poses_3d_array.shape[1])
            ]
        }
        
        # Save converted data
        with open(output_file, 'w') as f:
            json.dump(anatomical_data, f, indent=2)
        
        print(f"Successfully converted to {output_file}")
        print(f"Shape: {poses_3d_array.shape} (people, frames, keypoints, xyz)")
        
        return poses_3d_array, anatomical_data
        
    except Exception as e:
        print(f"Error converting file: {e}")
        return None, None

def main():
    """Main function to handle command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python temp_convert_h36m.py <input_h36m_file> [output_file]")
        print("Or: python temp_convert_h36m.py --create-sample")
        return
    
    # Convert provided file
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    poses_3d, anatomical_data = convert_h36m_to_anatomical(input_file, output_file)
    
    if poses_3d is not None:
        output_file = output_file or input_file.replace('.json', '_anatomical.json')
        print(f"\nConversion completed!")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")


if __name__ == "__main__":
    main() 