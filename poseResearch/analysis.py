import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def analyze_validation_results(
    csv_files, labels, output_name="motionbert_checkpoint_comparison"
):
    """
    Analyze validation results from CSV files and create a bar chart comparing keypoints across all YOLO models.

    Args:
        csv_files (list): List of paths to CSV files
        labels (list): List of labels for each validation run
        output_name (str): Base name for output files (without extension)
    """
    print(f"Looking for {len(csv_files)} validation files:")
    for f in csv_files:
        print(f"  {os.path.basename(f)}")

    # Read summary rows from each CSV file
    summary_data = []
    for i, file_path in enumerate(csv_files):
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Get the summary row (first row where frame_idx is 'SUMMARY')
                summary_row = df[df["frame_idx"] == "SUMMARY"].iloc[0]
                summary_data.append(summary_row)
                print(f"Loaded data for {labels[i]}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"Warning: File {file_path} not found")

    if not summary_data:
        print("No valid CSV files found!")
        return

    # Extract keypoint columns (excluding overall and confidence columns)
    keypoint_columns = []
    for col in summary_data[0].index:
        if (
            col.startswith("mpjpe_")
            and "overall" not in col
            and "confidence" not in col
            or col == "overall_mpjpe"
        ):
            keypoint_columns.append(col)

    # Create data for plotting
    keypoint_names = [
        col.replace("mpjpe_", "").replace("_", " ").title() for col in keypoint_columns
    ]
    keypoint_values = []

    for summary_row in summary_data:
        values = [summary_row[col] for col in keypoint_columns]
        keypoint_values.append(values)

    # Convert to numpy array for easier manipulation
    keypoint_values = np.array(keypoint_values)

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(18, 12))

    # Set up the bar positions
    x = np.arange(len(keypoint_names))
    width = 0.15  # Width of bars (adjusted for 5 models)
    multiplier = 0

    # Create bars for each validation run
    for i, (label, values) in enumerate(zip(labels, keypoint_values)):
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=label, alpha=0.8)
        multiplier += 1

    # Customize the chart
    ax.set_xlabel("Keypoints", fontsize=12)
    ax.set_ylabel("MPJPE (mm)", fontsize=12)
    ax.set_title(
        f"Keypoint MPJPE Comparison - All MotionBERT Configurations (Cam01)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xticks(x + width * 2)  # Center the x-tick labels
    ax.set_xticklabels(keypoint_names, rotation=45, ha="right")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (label, values) in enumerate(zip(labels, keypoint_values)):
        offset = width * i
        for j, v in enumerate(values):
            ax.text(j + offset, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    # Save the plot
    output_path = f"validation/validation-results/{output_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Display the plot
    plt.show()

    # Also create a summary table
    print("\n=== MPJPE Summary Table ===")
    summary_df = pd.DataFrame(keypoint_values, columns=keypoint_names, index=labels)
    print(summary_df.round(2))

    # Save summary table
    summary_csv_path = f"validation/validation-results/{output_name}_summary.csv"
    summary_df.to_csv(summary_csv_path)
    print(f"\nSummary table saved to: {summary_csv_path}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Analyze validation results from CSV files and create comparison charts"
    )

    parser.add_argument(
        "--csv-files",
        nargs="+",
        required=True,
        help="List of CSV file paths to analyze",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="List of labels for each validation run (must match number of CSV files)",
    )

    parser.add_argument(
        "--output-name",
        default="motionbert_checkpoint_comparison",
        help="Base name for output files (without extension)",
    )

    args = parser.parse_args()

    # Validate that we have the same number of CSV files and labels
    if len(args.csv_files) != len(args.labels):
        print(
            f"Error: Number of CSV files ({len(args.csv_files)}) must match number of labels ({len(args.labels)})"
        )
        return

    # Call the analysis function with parsed arguments
    analyze_validation_results(args.csv_files, args.labels, args.output_name)


if __name__ == "__main__":
    # For backward compatibility, if no arguments provided, use the hardcoded defaults
    import sys

    if len(sys.argv) == 1:
        # Hardcoded paths to the CSV files for all 5 YOLO models
        csv_files = [
            "validation/validation-results/validation-gt-none-yolo-11m-motionbert-global_lite-malemonologue2_t2-cam01-full--3d-none-yolo-11m-motionbert-global_lite-malemonologue2_t2-cam01-full.csv",
            "validation/validation-results/validation-gt-none-yolo-11m-motionbert-train_h36m-malemonologue2_t2-cam01-full--3d-none-yolo-11m-motionbert-train_h36m-malemonologue2_t2-cam01-full.csv",
            "validation/validation-results/validation-gt-none-yolo-11m-motionbert-ft_h36m-malemonologue2_t2-cam01-full--3d-none-yolo-11m-motionbert-ft_h36m-malemonologue2_t2-cam01-full.csv",
        ]

        # Labels for the different validation runs
        labels = ["global_lite", "train_h36m", "ft_h36m"]

        print("No arguments provided, using hardcoded defaults...")
        analyze_validation_results(csv_files, labels)
    else:
        main()
