import cv2
from cv2.typing import MatLike
import json
import torch
from pipeline import EstimationPipe
from estimation.preprocess.yolo_preprocess import YOLOPreprocess
from estimation.pose2D.yolo_estimation import YOLOEstimation
from estimation.pose3D.pose_estimation_3D import ThreeDPoseEstimation
from utils.output_saver import OutputSaver
from typing import Optional


class Dummy3DPose(ThreeDPoseEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "Dummy3DPose"

    def _forward(self, poses_2d: torch.Tensor) -> torch.Tensor:
        # Just return the input for test
        return poses_2d


class DummyOutputSaver(OutputSaver):
    def handle(self, output, config):
        pass


def resize_to_640_640(video_path: str, num_frames: Optional[int] = 5) -> list[MatLike]:
    """
    Resize a video to 640x640.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or num_frames is not None and count >= num_frames:
            break
        # Resize frame to target size (width, height)
        frame = cv2.resize(frame, (640, 640))
        # Convert BGR (OpenCV) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to float32 and normalize to [0, 1]
        frame = frame.astype("float32") / 255.0
        frames.append(frame)
        count += 1
    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames read from video.")
    return frames


def video_to_tensor(video_path: str, num_frames: int):
    """
    Convert a video to a tensor in BCHW format.
    Supported video formats: https://docs.ultralytics.com/de/modes/predict/#videos
    Args:
        video_path (str): Path to the video.
        num_frames (int): Number of frames to read from the video.
    Returns:
        torch.Tensor: Tensor of shape (T, H, W, C)
    """
    frames = resize_to_640_640(video_path, num_frames=num_frames)
    frames_tensor = torch.tensor(frames)
    return frames_tensor


def test_yolo_pipeline_stages(video_path: str, model_name: str, num_frames: int = 5):
    """
    Test the estimation classes NoPreprocess and YOLOEstimation.
    Args:
        video_path (str): Path to the video.
        num_frames (int): Number of frames to read from the video.
    """
    images = video_to_tensor(video_path, num_frames=num_frames)
    preprocess = YOLOPreprocess()
    images = preprocess.forward(images)
    print("After PreprocessEstimation:", images.shape)
    pose2d = YOLOEstimation(model_name)
    poses_2d = pose2d.forward(images)
    print("After 2DPoseEstimation:", poses_2d.shape)
    pose3d = Dummy3DPose()
    poses_3d = pose3d.forward(poses_2d)
    print("After 3DPoseEstimation:", poses_3d.shape)
    # save results to json
    filename = "results.json"
    with open(filename, "w") as f:
        json.dump(poses_3d.tolist(), f)


def test_yolo_pipeline(
    video_path: str,
    model_name: str,
    num_frames: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    """
    Test EstimationPipe with NoPreprocess and YOLOEstimation.
    Args:
        video_path (str): Path to the video.
        num_frames (int): Number of frames to read from the video.
    """
    if batch_size is None:
        batch_size = 5

    class SimpleDataLoader:
        def __init__(self, images_tensor: torch.Tensor, batch_size: int):
            self.images_tensor = images_tensor
            self.batch_size = batch_size

        def __iter__(self):
            # Yield batch_size images at a time
            # If there are fewer than batch_size images, yield what is available
            num_images = self.images_tensor.shape[0]
            for i in range(0, num_images, self.batch_size):
                yield self.images_tensor[i : i + self.batch_size]

        def __len__(self):
            num_images = images_tensor.shape[0]
            return (num_images + self.batch_size - 1) // self.batch_size

    preprocessor = YOLOPreprocess()
    images_tensor = preprocessor.input_video_to_tensor(
        video_path, num_frames=num_frames
    )

    dataloader = SimpleDataLoader(images_tensor, batch_size=batch_size)
    pipeline = EstimationPipe(
        preprocessor=preprocessor,
        flatpose=YOLOEstimation(model_name),
        poselifting=Dummy3DPose(),
        output_saver=DummyOutputSaver(),
    )
    results = []
    for output in pipeline.forward(dataloader):
        print("Pipeline output shape:", output.shape)
        results.append(output)
    # Save the last output to json
    if results:
        filename = "results.json"
        with open(filename, "w") as f:
            json.dump(results[-1].tolist(), f)


if __name__ == "__main__":
    video_path = "fem1_t1_preview.mp4"
    model_name = "yolo11s-pose.pt"
    # test_yolo_pipeline_stages(video_path, model_name, num_frames=5)
    test_yolo_pipeline(video_path, model_name, num_frames=5)
