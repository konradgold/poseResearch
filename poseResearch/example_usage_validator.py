#!/usr/bin/env python3
"""
Example usage of the simplified PoseValidator.

This script demonstrates how to:
1. Load DataLoaders from JSON files containing 2D and 3D poses
2. Use the PoseValidator to validate single sequences
3. Use batch validation for multiple sequences
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import DataLoader
from validation.pose_validator import PoseValidator


def example_single_validation():
    """Demonstrate single sequence validation."""
    print("\n" + "=" * 50)
    print("SINGLE SEQUENCE VALIDATION EXAMPLE")
    print("=" * 50)

    # Load ground truth DataLoader from JSON
    gt_loader = DataLoader()
    gt_loader.load_json("dataloader/results_flatpose.json")
    print("Loaded ground truth DataLoader from JSON")

    # Load 3D poses DataLoader from JSON
    poses_3d_loader = DataLoader()
    poses_3d_loader.load_json("dataloader/results_3d.json")
    print("Loaded 3D poses DataLoader from JSON")

    # Create validator
    validator = PoseValidator(confidence_threshold=0.0, image_size=(640, 480))

    # Validate
    print("\nRunning validation...")
    similarity_score = validator.validate(
        gt_data_loader=gt_loader,
        poses_3d_data_loader=poses_3d_loader,
        gt_stage="flatpose",
        poses_3d_stage="poselifting",
    )

    print(f"Similarity Score: {similarity_score:.4f}")

    return similarity_score


def example_batch_validation():
    """Demonstrate batch validation with multiple sequences."""
    print("\n" + "=" * 50)
    print("BATCH VALIDATION EXAMPLE")
    print("=" * 50)

    # Create multiple data files
    gt_loaders = []
    poses_3d_loaders = []

    num_sequences = 3
    for i in range(num_sequences):
        print(f"Creating data for sequence {i+1}/{num_sequences}...")

        # Create different sized data for each sequence
        num_frames = 20 + i * 10

        # Ground truth data
        gt_poses = torch.rand(2, num_frames, 17, 3)
        gt_poses[..., 0] *= 640
        gt_poses[..., 1] *= 480
        gt_poses[..., 2] = torch.rand(2, num_frames, 17) * 0.4 + 0.6

        gt_filename = f"sample_gt_poses_{i}.json"
        gt_data = {
            "flatpose": {
                "data": gt_poses.tolist(),
                "shape": list(gt_poses.shape),
                "config": {"stage_name": "flatpose", "description": f"GT sequence {i}"},
            }
        }
        with open(gt_filename, "w") as f:
            json.dump(gt_data, f)

        # 3D poses data
        poses_3d = torch.randn(2, num_frames, 17, 3)
        poses_3d[..., 2] = torch.abs(poses_3d[..., 2]) + 1.0

        poses_3d_filename = f"sample_3d_poses_{i}.json"
        poses_3d_data = {
            "poselifting": {
                "data": poses_3d.tolist(),
                "shape": list(poses_3d.shape),
                "config": {
                    "stage_name": "poselifting",
                    "description": f"3D sequence {i}",
                },
            }
        }
        with open(poses_3d_filename, "w") as f:
            json.dump(poses_3d_data, f)

        # Load DataLoaders
        gt_loader = DataLoader()
        gt_loader.load_json(gt_filename)
        gt_loaders.append(gt_loader)

        poses_3d_loader = DataLoader()
        poses_3d_loader.load_json(poses_3d_filename)
        poses_3d_loaders.append(poses_3d_loader)

    # Create validator and run batch validation
    validator = PoseValidator(confidence_threshold=0.4)

    print(f"\nRunning batch validation on {num_sequences} sequences...")
    batch_scores = validator.validate_batch(
        gt_data_loaders=gt_loaders,
        poses_3d_data_loaders=poses_3d_loaders,
        gt_stage="flatpose",
        poses_3d_stage="poselifting",
    )

    # Print results
    print("\nBatch Validation Results:")
    for i, score in enumerate(batch_scores):
        print(f"  Sequence {i+1}: {score:.4f}")

    avg_score = np.mean(batch_scores)
    print(f"\nAverage Similarity Score: {avg_score:.4f}")

    # Clean up temporary files
    for i in range(num_sequences):
        Path(f"sample_gt_poses_{i}.json").unlink(missing_ok=True)
        Path(f"sample_3d_poses_{i}.json").unlink(missing_ok=True)

    return batch_scores


def main():
    """Run all examples."""
    print("POSE VALIDATOR - EXAMPLE USAGE")
    print("=" * 50)

    try:

        # Single validation example
        single_score = example_single_validation()

        # Batch validation example
        # batch_scores = example_batch_validation()

        # Summary
        print("\n" + "=" * 50)
        print("EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 50)

        print(f"\nResults Summary:")
        print(f"Single validation score: {single_score:.4f}")
        # print(f"Batch validation average: {np.mean(batch_scores):.4f}")

        # print(f"\nBatch individual scores: {[f'{s:.4f}' for s in batch_scores]}")

    except Exception as e:
        print(f"Error during example execution: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Clean up main sample files
        Path("sample_gt_poses.json").unlink(missing_ok=True)
        Path("sample_3d_poses.json").unlink(missing_ok=True)
        print("Cleaned up temporary files")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
