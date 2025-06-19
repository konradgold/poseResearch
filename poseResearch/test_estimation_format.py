import torch
from estimation.preprocess.preprocess_estimation import PreprocessEstimation
from estimation.pose2D.pose_estimation_2D import TwoDPoseEstimation
from estimation.pose3D.pose_estimation_3D import ThreeDPoseEstimation


# Dummy implementations for testing
class DummyPreprocess(PreprocessEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "DummyPreprocess"

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        # Just return the input
        return images


class Dummy2DPose(TwoDPoseEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "Dummy2DPose"

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        # Simulate 2D pose output: (P, T, Nk, D)
        T, H, W, C = images.shape
        P = 1
        Nk = getattr(self, "num_keypoints", 17)
        D = getattr(self, "num_dims", 3)
        return torch.zeros((P, T, Nk, D))


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

    preprocess = DummyPreprocess()
    pose2d = Dummy2DPose()
    pose3d = Dummy3DPose()

    out_pre = preprocess.forward(images)
    print("After PreprocessEstimation:", out_pre.shape)
    out_2d = pose2d.forward(out_pre)
    print("After 2DPoseEstimation:", out_2d.shape)
    out_3d = pose3d.forward(out_2d)
    print("After 3DPoseEstimation:", out_3d.shape)
