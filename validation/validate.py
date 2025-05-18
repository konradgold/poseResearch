import argparse
import pickle
from dataset.dataloader import VideoHandler, CameraAngle
from validation.pose_validation import PoseValidation
from validation.plane_pose_estimation import PlanePoseEstimator

def main():
    parser = argparse.ArgumentParser(description="Validate video with pickle data.")
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--validate', type=str, required=True, help='Path to the .pkl file')
    parser.add_argument('--results', type=str, required=True, help='Path to the .pkl file')
    args = parser.parse_args()

    with open(args.validate, 'rb') as f:
        poses = pickle.load(f)
    print("Loaded pickle data:", type(poses))

    video = VideoHandler(args.video)

    ground_truth = PlanePoseEstimator().get_pose(video.get_angle_stream(CameraAngle.top_view))

    validator = PoseValidation(
        ground_truth, 
        poses,
        args.results)
    validator.average_similarity()


if __name__ == "__main__":
    main()