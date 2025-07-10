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

    def _post_process(self, output: torch.Tensor) -> torch.Tensor:
        """
        Post-process the output from _forward method.
        This method is called automatically after _forward and before output validation.
        Override this method to implement custom post-processing (e.g., format conversion).

        Args:
            output (torch.Tensor): Output from _forward method

        Returns:
            torch.Tensor: Post-processed output
        """
        return output

    def _normalization(self, output: torch.Tensor) -> torch.Tensor:
        """
        Normalize the output by centering root at origin and scaling root-belly distance to 1.
        This method is called automatically after _post_process.
        Override this method to implement custom normalization.

        Args:
            output (torch.Tensor): Output from _post_process method

        Returns:
            torch.Tensor: Normalized output
        """
        # Default implementation: return unchanged
        return output

    def forward(self, batch) -> torch.Tensor:
        output = self._forward(batch)
        output = self._post_process(output)
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
