from abc import ABC, abstractmethod
from typing import Any, Dict
from torch.optim import Optimizer
import torch


class VQVAEBase(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Dict[str, Any]: Output dictionary containing at least:
                - 'recon': reconstructed input
                - 'loss': total loss
                - optionally: 'vq_loss', 'perplexity', 'z_e', etc.
        """
        pass

    @abstractmethod
    def train_step(
        self, batch: torch.Tensor, optimizer: Optimizer, scheduler: Any = None
    ) -> Dict[str, float]:
        """
        Performs a single training step.

        Args:
            batch (torch.Tensor): A batch of training data.
            optimizer (Optimizer): Optimizer instance.
            scheduler (Optional): Learning rate scheduler.

        Returns:
            Dict[str, float]: Logging info like total loss, VQ loss, recon loss, etc.
        """
        pass

    @abstractmethod
    def load_parameters(self, path: str, strict: bool = True) -> None:
        """
        Loads model parameters from a checkpoint.

        Args:
            path (str): Path to the checkpoint.
            strict (bool): Whether to strictly enforce that the keys match.
        """
        pass
