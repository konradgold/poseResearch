import torch
from estimation.preprocess.no_preprocess import NoPreprocess
from estimation.pose2D.yolo_estimation import YOLOEstimation
from estimation.pose3D.pose_estimation_3D import ThreeDPoseEstimation


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


if __name__ == "__main__":
    # Create dummy input: (T, H, W, C)
    T, H, W, C = 5, 64, 64, 3
    images = torch.randn(T, H, W, C)

    preprocess = NoPreprocess()
    pose2d = YOLOEstimation("yolo11n-pose.pt")
    pose3d = Dummy3DPose()

    out_pre = preprocess.forward(images)
    print("After PreprocessEstimation:", out_pre.shape)
    out_2d = pose2d.forward(out_pre)
    print("After 2DPoseEstimation:", out_2d.shape)
    out_3d = pose3d.forward(out_2d)
    print("After 3DPoseEstimation:", out_3d.shape)
