from .preprocess_estimation import PreprocessEstimation
import torch


class NoPreprocess(PreprocessEstimation):
    @property
    def config(self):
        return {}

    @property
    def identifier(self):
        return "NoPreprocess"

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        return images
