#!/usr/bin/env python3
"""
Temporary conversion script to convert pose JSON format to DataLoader format.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List


def convert_pose_json_to_dataloader_format(
    input_file: str, output_file: str, stage_name: str = "poselifting"
) -> None:
    """
    Convert pose estimation JSON to DataLoader format.

    Args:
        input_file: Path to input JSON file with pose data
        output_file: Path to output JSON file in DataLoader format
        stage_name: Name of the pipeline stage ("preprocessor", "flatpose", "poselifting")
    """

    # Read the input JSON
    with open(input_file, "r") as f:
        data = json.load(f)

    # Extract pose data
    poses_3d = data.get("poses_3d", [])
    metadata = data.get("metadata", {})

    print(f"Input data structure:")
    print(f"  poses_3d length: {len(poses_3d)}")
    if poses_3d:
        print(f"  First element length: {len(poses_3d[0])}")
        if poses_3d[0]:
            print(f"  First frame length: {len(poses_3d[0][0])}")
            if poses_3d[0][0]:
                print(f"  First keypoint: {poses_3d[0][0][0]}")

    # Convert to numpy array to understand the structure better
    if poses_3d and len(poses_3d) > 0:
        poses_array = np.array(poses_3d[0])  # Take the first (and likely only) sequence
        print(f"  Converted shape: {poses_array.shape}")

        # The data appears to be (frames, keypoints, coordinates)
        # We need to add a "people" dimension to match expected format
        if len(poses_array.shape) == 3:
            # Add people dimension: (frames, keypoints, coords) -> (people, frames, keypoints, coords)
            poses_array = poses_array[np.newaxis, ...]  # Add people dimension
            print(f"  After adding people dimension: {poses_array.shape}")

        # Convert to expected format: (people, frames, keypoints, coordinates)
        final_shape = poses_array.shape

        # Create DataLoader format
        dataloader_format = {
            stage_name: {
                "data": poses_array.tolist(),
                "shape": list(final_shape),
                "config": {
                    "stage_name": stage_name,
                    "converted_from": input_file,
                    "original_metadata": metadata,
                },
            }
        }

        # Save the converted data
        with open(output_file, "w") as f:
            json.dump(dataloader_format, f, indent=2)

        print(f"Converted data saved to: {output_file}")
        print(f"Final shape: {final_shape}")
        print(f"Stage name: {stage_name}")

    else:
        print("Error: No pose data found in input file")


def batch_convert_files(
    input_pattern: str, output_dir: str = "converted_dataloader"
) -> None:
    """
    Convert multiple files matching a pattern.

    Args:
        input_pattern: Glob pattern for input files (e.g., "results_*.json")
        output_dir: Directory to save converted files
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    input_files = list(Path(".").glob(input_pattern))

    for input_file in input_files:
        # Determine output filename
        output_filename = f"dataloader_{input_file.stem}.json"
        output_file = output_path / output_filename

        # Determine stage name based on filename or default to poselifting
        if "3d" in input_file.name.lower() or "lifting" in input_file.name.lower():
            stage_name = "poselifting"
        elif "2d" in input_file.name.lower() or "flat" in input_file.name.lower():
            stage_name = "flatpose"
        else:
            stage_name = "poselifting"  # Default for 3D poses

        print(f"\nConverting: {input_file} -> {output_file}")
        convert_pose_json_to_dataloader_format(
            str(input_file), str(output_file), stage_name
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  python temp_convert_dataloader.py <input_file> [output_file] [stage_name]"
        )
        print("  python temp_convert_dataloader.py --batch <pattern>")
        print("")
        print("Examples:")
        print(
            "  python temp_convert_dataloader.py results_interactive4_t3-cam16_anatomical.json"
        )
        print(
            "  python temp_convert_dataloader.py results_3d.json output.json poselifting"
        )
        print("  python temp_convert_dataloader.py --batch 'results_*.json'")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        pattern = sys.argv[2] if len(sys.argv) > 2 else "results_*.json"
        batch_convert_files(pattern)
    else:
        input_file = sys.argv[1]
        output_file = (
            sys.argv[2]
            if len(sys.argv) > 2
            else f"dataloader_{Path(input_file).stem}.json"
        )
        stage_name = sys.argv[3] if len(sys.argv) > 3 else "poselifting"

        convert_pose_json_to_dataloader_format(input_file, output_file, stage_name)
