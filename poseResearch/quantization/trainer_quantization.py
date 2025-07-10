import yaml
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from utils.process_manager import ProcessManager
from tqdm import tqdm
from poseResearch.quantization.base_quantizer import VQVAEBase


class VQVAETrainer:
    def __init__(
        self,
        model: VQVAEBase,
        train_loader: ProcessManager,
        optimizer: torch.optim.Optimizer,
        val_loader: Optional[ProcessManager] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config_path: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.config = self._load_config(config_path) if config_path else {}

        self.epochs = self.config.get("epochs", 10)
        self.log_interval = self.config.get("log_interval", 10)
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")
            self._train_epoch(epoch)
            if self.val_loader:
                self._eval_epoch(epoch)
            self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()
            metrics = self.model.train_step(batch, self.optimizer, self.scheduler)
            loss = metrics.get("loss", 0.0)

            running_loss += loss
            if self.scheduler:
                self.scheduler.step()

            if batch_idx % self.log_interval == 0:
                print(f"  [Batch {batch_idx}] Loss: {loss:.4f}")

        avg_loss = running_loss / len(self.train_loader)
        print(f"==> Train Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

    def _eval_epoch(self, epoch: int):
        assert (
            self.val_loader is not None
        ), "Validation loader must be provided for evaluation."
        print(f"Validating Epoch {epoch}...")
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = batch.to(self.device)
                outputs = self.model.forward(batch)
                total_loss += outputs.get("loss", 0.0)

        avg_loss = total_loss / len(self.val_loader)
        print(f"==> Val Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

    def _save_checkpoint(self, epoch: int):
        path = self.checkpoint_dir / f"vqvae_epoch_{epoch}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": (
                    self.optimizer.state_dict() if self.optimizer else None
                ),
                "epoch": epoch,
            },
            path,
        )
        print(f"Checkpoint saved to {path}")
