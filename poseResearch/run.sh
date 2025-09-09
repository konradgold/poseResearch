# Example usage of the pipeline + visualization
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the pipeline with YOLOBoundingBoxPreprocessor, YOLO2dEstimator, MotionBert3dEstimator
python "$SCRIPT_DIR/example_usage.py" 10 --preprocessor yolo_bb --pose2d yolo --pose3d motionbert --yolo-model 11m --video male1_t4_preview.mp4 --batch-size 20
# Visualize the results
python "$SCRIPT_DIR/example_visualize.py" --path yolo_bb-yolo-motionbert-11m-male1_t4_preview-full
# Validate 3D poses against their own 2D poses
python "$SCRIPT_DIR/example_usage_validator.py" --gt-name yolo_bb-yolo-motionbert-11m-male1_t4_preview-full --poses3d-name yolo_bb-yolo-motionbert-11m-male1_t4_preview-full



# Temporary
# none-yolo-motionbert-11m-malemonologue2_t2-cam01-full
python example_usage_validator.py --gt-name none-yolo-motionbert-11m-malemonologue2_t2-cam01-full --poses3d-name none-yolo-motionbert-11m-malemonologue2_t2-cam01-full
# Cam 01 - yolo_bb preprocessing
python example_usage.py 10 --preprocessor yolo_bb --pose2d yolo --pose3d motionbert --yolo-model 11m --video malemonologue2_t2-cam01.mp4 --batch-size 20
python example_visualize.py --path yolo_bb-yolo-motionbert-11m-malemonologue2_t2-cam01-full
python example_usage_validator.py --gt-name yolo_bb-yolo-motionbert-11m-malemonologue2_t2-cam01-full --poses3d-name yolo_bb-yolo-motionbert-11m-malemonologue2_t2-cam01-full
# Cam 22 - no preprocessing
python example_usage.py 10 --preprocessor none --pose2d yolo --pose3d motionbert --yolo-model 11m --video malemonologue2_t2-cam22.mp4 --batch-size 20
python example_visualize.py --path none-yolo-motionbert-11m-malemonologue2_t2-cam22-full
python example_usage_validator.py --gt-name none-yolo-motionbert-11m-malemonologue2_t2-cam22-full --poses3d-name none-yolo-motionbert-11m-malemonologue2_t2-cam22-full
# Cam 22 - yolo_bb preprocessing
python example_usage.py 10 --preprocessor yolo_bb --pose2d yolo --pose3d motionbert --yolo-model 11m --video malemonologue2_t2-cam22.mp4 --batch-size 20
python example_visualize.py --path yolo_bb-yolo-motionbert-11m-malemonologue2_t2-cam22-full
python example_usage_validator.py --gt-name yolo_bb-yolo-motionbert-11m-malemonologue2_t2-cam22-full --poses3d-name yolo_bb-yolo-motionbert-11m-malemonologue2_t2-cam22-full

python example_usage_validator.py --gt-name none-yolo-motionbert-11m-malemonologue2_t2-cam01-full --poses3d-name none-yolo-motionbert-11m-malemonologue2_t2-cam22-full