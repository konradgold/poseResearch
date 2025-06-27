from abc import ABC, abstractmethod
import torch


class Estimation(ABC):
    """
    Abstract base class for estimation in the pipeline.
    To implement a new estimation, inherit from this class and implement the following methods:
    - _forward: forward pass of the estimation
    - output_check: check if the output is valid
    - config: configuration of the estimation
    - identifier: identifier of the estimation
    - forward: forward pass of the estimation
    - output_check: check if the output is valid
    """

    @abstractmethod
    def _forward(self, batch) -> torch.Tensor:
        pass

    def forward(self, batch) -> torch.Tensor:
        output = self._forward(batch)
        if self.output_check(output):
            print(
                f"Forward of {self.identifier} is done with output shape {output.shape}."
            )
            return output
        else:
            raise RuntimeError(f"{self.identifier} did not return expected output.")

    @abstractmethod
    def output_check(self, output) -> bool:
        pass

    @property
    @abstractmethod
    def config(self) -> dict:
        pass

    @property
    @abstractmethod
    def identifier(self) -> str:
        pass
