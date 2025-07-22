#!/usr/bin/env python3
"""
Example usage of the simplified PoseValidator.

This script demonstrates how to:
1. Load DataLoaders from JSON files containing 2D and 3D poses
2. Use the PoseValidator to validate single sequences
3. Use batch validation for multiple sequences
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from utils.process_manager import ProcessManager
from validation.pose_validator import PoseValidator


def example_single_validation(gt_name: str, poses3d_name: str):
    """Demonstrate single sequence validation."""
    print("\n" + "=" * 50)
    print("SINGLE SEQUENCE VALIDATION EXAMPLE")
    print("=" * 50)

    pr_dir = Path(__file__).parent

    # Load ground truth DataLoader from JSON
    gt_loader = ProcessManager()
    gt_loader.load_json(pr_dir / "dataloader" / f"results_{gt_name}_flatpose.json")
    print(f"Loaded ground truth DataLoader from: {gt_name}")

    # Load 3D poses DataLoader from JSON
    poses_3d_loader = ProcessManager()
    poses_3d_loader.load_json(
        pr_dir / "dataloader" / f"results_{poses3d_name}_poselifting.json"
    )
    print(f"Loaded 3D poses DataLoader from: {poses3d_name}")

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run pose validation examples with custom data paths."
    )
    parser.add_argument(
        "--gt-name",
        type=str,
        help="Path to ground truth JSON file (2D poses)",
        default="male2_t2_cam22",
    )
    parser.add_argument(
        "--poses3d-name",
        type=str,
        help="Path to 3D poses JSON file",
        default="male2_t2_cam01",
    )
    return parser.parse_args()


def main():
    """Run all examples."""
    print("POSE VALIDATOR - EXAMPLE USAGE")
    print("=" * 50)

    args = parse_args()

    try:
        # Single validation example
        single_score = example_single_validation(args.gt_name, args.poses3d_name)

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
