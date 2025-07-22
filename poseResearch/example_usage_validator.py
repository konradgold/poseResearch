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

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from utils.process_manager import ProcessManager
from validation.pose_validator import PoseValidator


def example_single_validation():
    """Demonstrate single sequence validation."""
    print("\n" + "=" * 50)
    print("SINGLE SEQUENCE VALIDATION EXAMPLE")
    print("=" * 50)

    # Load ground truth DataLoader from JSON
    gt_loader = ProcessManager()
    gt_loader.load_json("dataloader/male2_t2_cam22/results_flatpose.json")
    print("Loaded ground truth DataLoader from JSON")

    # Load 3D poses DataLoader from JSON
    poses_3d_loader = ProcessManager()
    poses_3d_loader.load_json("dataloader/male2_t2_cam01/results_poselifting.json")
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

    return similarity_score


def main():
    """Run all examples."""
    print("POSE VALIDATOR - EXAMPLE USAGE")
    print("=" * 50)

    try:

        # Single validation example
        single_score = example_single_validation()

        print(f"Validation score: {single_score:.4f}")

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
