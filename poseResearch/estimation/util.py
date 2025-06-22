from abc import ABC, abstractmethod
import torch


class Estimation(ABC):

    @abstractmethod
    def _forward(self, batch) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        output = self._forward(batch)
        if self.output_check(output):
            return output
        else:
            raise RuntimeError(f"{self.identifier} did not return expected output")

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
