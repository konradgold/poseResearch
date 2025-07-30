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

    Optional methods to override:
    - _pre_process: pre-process input before forward pass
    - _normalization: normalize output after forward pass
    """

    @abstractmethod
    def _forward(self, batch) -> torch.Tensor:
        pass

    def _pre_process(self, batch) -> torch.Tensor:
        """
        Pre-process the input before _forward method.
        This method is called automatically before _forward.
        Override this method to implement custom pre-processing (e.g., format conversion).

        Args:
            batch: Input batch to be pre-processed

        Returns:
            torch.Tensor: Pre-processed input
        """
        return batch

    def _normalization(self, output: torch.Tensor) -> torch.Tensor:
        """
        Normalize the output by centering root at origin and scaling root-belly distance to 1.
        This method is called automatically after _forward.
        Override this method to implement custom normalization.

        Args:
            output (torch.Tensor): Output from _forward method

        Returns:
            torch.Tensor: Normalized output
        """
        # Default implementation: return unchanged
        return output

    def forward(self, batch) -> torch.Tensor:
        preprocessed_batch = self._pre_process(batch)
        output = self._forward(preprocessed_batch)
        output = self._normalization(output)
        print(f"{self.identifier} forwarded output of shape: {output.shape}.")
        if self.output_check(output):
            print(f"Output of {self.identifier} is valid with shape {output.shape}.")
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
