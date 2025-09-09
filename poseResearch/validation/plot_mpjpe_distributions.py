import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_mpjpe_distributions(csv_file_path):
    """
    Read MPJPE data from CSV and create box plots showing distributions for each joint.

    Args:
        csv_file_path (str): Path to the CSV file containing MPJPE data
    """

    # Read the CSV file, skipping the SUMMARY row
    df = pd.read_csv(csv_file_path, skiprows=[1])

    # Get the MPJPE columns (excluding frame_idx, confidence_threshold, and real_mpjpe columns)
    mpjpe_columns = [
        col
        for col in df.columns
        if col.startswith("mpjpe_") and not col.startswith("mpjpe_real")
    ]

    # Create a clean dataset with only MPJPE values
    mpjpe_data = df[mpjpe_columns].copy()

    # Rename columns to be more readable (remove 'mpjpe_' prefix)
    mpjpe_data.columns = [col.replace("mpjpe_", "") for col in mpjpe_data.columns]

    # Set up the plotting style
    plt.style.use("default")

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create box plot with better colors
    box_plot = mpjpe_data.boxplot(
        ax=ax,
        grid=False,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="darkblue", linewidth=1.5),
        capprops=dict(color="darkblue", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="orange", markersize=4),
    )

    # Customize the plot
    ax.set_title(
        "Distribution of MPJPE Values by Body Joint (ft_h36m)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Body Joints", fontsize=12, fontweight="bold")
    ax.set_ylabel("MPJPE (mm)", fontsize=12, fontweight="bold")

    # Rotate x-axis labels for better readability
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis="y")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    output_path = Path(csv_file_path).parent / "mpjpe_distributions_boxplot_ft_h36m.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Box plot saved to: {output_path}")

    # Show the plot
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    summary_stats = mpjpe_data.describe()
    print(summary_stats)

    # Print the joints with highest and lowest median MPJPE
    medians = mpjpe_data.median().sort_values(ascending=False)
    print(f"\nJoints with highest median MPJPE:")
    print(medians.head(5))
    print(f"\nJoints with lowest median MPJPE:")
    print(medians.tail(5))

    return mpjpe_data


def create_detailed_analysis(csv_file_path):
    """
    Create additional detailed analysis plots and statistics.

    Args:
        csv_file_path (str): Path to the CSV file containing MPJPE data
    """

    # Read the CSV file, skipping the SUMMARY row
    df = pd.read_csv(csv_file_path, skiprows=[1])

    # Get the MPJPE columns
    mpjpe_columns = [
        col
        for col in df.columns
        if col.startswith("mpjpe_") and not col.startswith("mpjpe_real")
    ]

    # Create a clean dataset with only MPJPE values
    mpjpe_data = df[mpjpe_columns].copy()

    # Rename columns to be more readable
    mpjpe_data.columns = [col.replace("mpjpe_", "") for col in mpjpe_data.columns]

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle("Detailed MPJPE Analysis (ft_h36m)", fontsize=18, fontweight="bold")

    # 1. Violin plot
    ax1 = axes[0, 0]
    # Prepare data for violin plot (melt the dataframe)
    mpjpe_melted = mpjpe_data.melt(var_name="Joint", value_name="MPJPE")
    sns.violinplot(data=mpjpe_melted, x="Joint", y="MPJPE", ax=ax1)
    ax1.set_title("MPJPE Distribution by Joint (Violin Plot)", fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # 2. Histogram of overall MPJPE
    ax2 = axes[0, 1]
    overall_mpjpe = df["overall_mpjpe"]
    ax2.hist(overall_mpjpe, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    ax2.set_title("Distribution of Overall MPJPE", fontweight="bold")
    ax2.set_xlabel("Overall MPJPE (mm)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    # 3. Heatmap of correlation between joints
    ax3 = axes[1, 0]
    correlation_matrix = mpjpe_data.corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax3,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )
    ax3.set_title("Correlation Between Joint MPJPE Values", fontweight="bold")

    # 4. Time series of overall MPJPE
    ax4 = axes[1, 1]
    ax4.plot(df.index, overall_mpjpe, linewidth=1, alpha=0.8)
    ax4.set_title("Overall MPJPE Over Time (Frames)", fontweight="bold")
    ax4.set_xlabel("Frame Index")
    ax4.set_ylabel("Overall MPJPE (mm)")
    ax4.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save the detailed analysis
    output_path = Path(csv_file_path).parent / "mpjpe_detailed_analysis_ft_h36m.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Detailed analysis saved to: {output_path}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Path to the CSV file
    csv_file_path = "validation-results/validation-gt-none-yolo-11m-motionbert-ft_h36m-malemonologue2_t2-cam01-full--3d-none-yolo-11m-motionbert-ft_h36m-malemonologue2_t2-cam01-full.csv"

    try:
        # Create the main box plot
        print("Creating MPJPE distribution box plots...")
        mpjpe_data = plot_mpjpe_distributions(csv_file_path)

        # Create detailed analysis
        print("\nCreating detailed analysis plots...")
        create_detailed_analysis(csv_file_path)

        print("\nAnalysis complete!")

    except FileNotFoundError:
        print(f"Error: Could not find the CSV file: {csv_file_path}")
        print("Please make sure the file path is correct.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check the CSV file format and try again.")
