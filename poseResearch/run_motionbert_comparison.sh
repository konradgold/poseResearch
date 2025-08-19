#!/bin/bash

# MotionBERT Configuration Comparison Pipeline
# This script runs the pipeline with different MotionBERT configurations and analyzes the results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_FILE="poseResearch/malemonologue2_t2-cam01.mp4"  # Using Cam01 video for consistency
BATCH_SIZE=20

echo "=== MotionBERT Configuration Comparison Pipeline ==="
echo "Video: $VIDEO_FILE"
echo "Batch Size: $BATCH_SIZE"
echo "YOLO Model: 11m (fixed)"
echo ""

# Array of MotionBERT configurations to test
MOTIONBERT_CONFIGS=("global_lite" "train_h36m" "ft_h36m")

# Run pipeline for each MotionBERT configuration
for config in "${MOTIONBERT_CONFIGS[@]}"; do
    echo "=== Processing with MotionBERT $config ==="
    
    # Run the pipeline
    python "$SCRIPT_DIR/example_usage.py" 10 \
        --preprocessor none \
        --pose2d yolo \
        --pose3d motionbert \
        --yolo-model "11m" \
        --config-name "$config" \
        --checkpoint-name "$config" \
        --video "$VIDEO_FILE" \
        --batch-size "$BATCH_SIZE"
    
    # Visualize the results
    echo "Visualizing results for MotionBERT $config..."
    python "$SCRIPT_DIR/example_visualize.py" --path "none-yolo-11m-motionbert-$config-malemonologue2_t2-cam01-full"
    
    # Validate 3D poses against their own 2D poses
    echo "Validating results for MotionBERT $config..."
    python "$SCRIPT_DIR/example_usage_validator.py" \
        --gt-name "none-yolo-11m-motionbert-$config-malemonologue2_t2-cam01-full" \
        --poses3d-name "none-yolo-11m-motionbert-$config-malemonologue2_t2-cam01-full"
    
    echo "Completed MotionBERT $config"
    echo ""
done

echo "=== All MotionBERT configurations processed ==="
echo ""

# Now analyze the results and create comparison plots
echo "=== Analyzing Results ==="
python "$SCRIPT_DIR/analysis.py" \
    --csv-files \
        "validation/validation-results/validation-gt-none-yolo-11m-motionbert-global_lite-malemonologue2_t2-cam01-full--3d-none-yolo-11m-motionbert-global_lite-malemonologue2_t2-cam01-full.csv" \
        "validation/validation-results/validation-gt-none-yolo-11m-motionbert-train_h36m-malemonologue2_t2-cam01-full--3d-none-yolo-11m-motionbert-train_h36m-malemonologue2_t2-cam01-full.csv" \
        "validation/validation-results/validation-gt-none-yolo-11m-motionbert-ft_h36m-malemonologue2_t2-cam01-full--3d-none-yolo-11m-motionbert-ft_h36m-malemonologue2_t2-cam01-full.csv" \
    --labels "global_lite" "train_h36m" "ft_h36m" \
    --output-name "motionbert_checkpoint_comparison"

echo "=== Analysis Complete ==="
echo "Check the generated plots in validation/validation-results/"
