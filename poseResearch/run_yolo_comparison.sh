#!/bin/bash

# YOLO Model Comparison Pipeline
# This script runs the pipeline with all 5 available YOLO models and analyzes the results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_FILE="malemonologue2_t2-cam22.mp4"  # Using Cam22 video for consistency
BATCH_SIZE=20

echo "=== YOLO Model Comparison Pipeline ==="
echo "Video: $VIDEO_FILE"
echo "Batch Size: $BATCH_SIZE"
echo ""

# Array of YOLO models to test
YOLO_MODELS=("11n" "11s" "11m" "11l" "11x")

# Run pipeline for each YOLO model
for model in "${YOLO_MODELS[@]}"; do
    echo "=== Processing with YOLO $model ==="
    
    # Run the pipeline
    python "$SCRIPT_DIR/example_usage.py" 10 \
        --preprocessor none \
        --pose2d yolo \
        --pose3d motionbert \
        --yolo-model "$model" \
        --video "$VIDEO_FILE" \
        --batch-size "$BATCH_SIZE"
    
    # Visualize the results
    echo "Visualizing results for YOLO $model..."
    python "$SCRIPT_DIR/example_visualize.py" --path "none-yolo-motionbert-$model-malemonologue2_t2-cam22-full"
    
    # Validate 3D poses against their own 2D poses
    echo "Validating results for YOLO $model..."
    python "$SCRIPT_DIR/example_usage_validator.py" \
        --gt-name "none-yolo-motionbert-$model-malemonologue2_t2-cam22-full" \
        --poses3d-name "none-yolo-motionbert-$model-malemonologue2_t2-cam22-full"
    
    echo "Completed YOLO $model"
    echo ""
done

echo "=== All YOLO models processed ==="
echo ""

# Now analyze the results and create comparison plots
echo "=== Analyzing Results ==="
python "$SCRIPT_DIR/analysis.py"

echo "=== Analysis Complete ==="
echo "Check the generated plots in validation/validation-results/"
