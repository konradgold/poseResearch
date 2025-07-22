# Example usage of the pipeline + visualization
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/example_usage.py" 10 --preprocessor yolo_bb --pose2d yolo --pose3d motionbert --yolo-model 11m --video "$SCRIPT_DIR/male1_t4_preview.mp4" --batch-size 20 --num-frames 20
python "$SCRIPT_DIR/example_visualize.py" --path yolo_bb-yolo-motionbert-11m-male1_t4_preview-20