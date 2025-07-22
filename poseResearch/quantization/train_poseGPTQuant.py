import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from poseResearch.quantization.poseGPT.transformer_vqvae import TransformerVQVAE
from tqdm import tqdm


class Data(torch.utils.data.DataLoader):
    def __init__(self, data: torch.Tensor, batch_size: int = 4, shuffle: bool = True):
        dataset = torch.utils.data.TensorDataset(data)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)



class TransformerVQVAETrainer:
    def __init__(
        self,
        model: TransformerVQVAE,
        train_loader: Data,
        optimizer: str = "adamw",
        val_loader: Optional[Data] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config_path: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = self._load_config(config_path) if config_path else {}
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=self.config.get("learning_rate", 1e-3)
            )
        elif optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.config.get("learning_rate", 1e-3)
            )
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=self.config.get("learning_rate", 1e-3)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        self.scheduler = scheduler
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

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
            batch = batch[0].to(self.device)

            self.optimizer.zero_grad()
            z_q, loss, indices = self.model.quantize(batch)
            print(f"Loss: {loss, loss.size()}")
            loss = loss.mean()  # Ensure loss is a scalar
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            



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
                outputs, loss, indices = self.model.quantize(batch)
                total_loss += loss

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a VQVAE model.")
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument(
        "--optimizer", type=str, choices=["adam", "adamw", "sgd"], default="adam"
    )
    args = parser.parse_args()

    # Assuming model and data loaders are defined elsewhere
    seq_len = 8
    model = TransformerVQVAE(in_dim = 51, n_layers = [2,2], hid_dim=64, heads=4, dropout=0.1, causal_encoder=True, causal_decoder=True, n_codebook=1, n_e=128, e_dim=51, beta=1.0, seq_len=seq_len)  
    with open("poseResearch/dataloader/results_3d.json", 'r') as f:
        data = json.load(f)
    data = data["poselifting"]["data"]
    poses = torch.Tensor(data).squeeze()
    if len(poses.size()) == 4:
        poses = poses.flatten(start_dim=0, end_dim=1)
    assert len(poses.size()) == 3
    assert poses.size(-2) == 17
    assert poses.size(-1) == 3
    assert poses.size(0) > 8
    poses = poses.permute(0, 2, 1).flatten(start_dim=1)
    poses = poses[:-(poses.size(0) % seq_len)]  # Ensure divisible by seq_len
    poses = poses.view(-1, seq_len, poses.size(-1))

    print(f"Input shape: {poses.shape}")

    train_loader = Data(poses, batch_size=32, shuffle=True)
    val_loader = None  # Replace with actual validation data loader if available

    trainer = TransformerVQVAETrainer(
        model=model,
        train_loader=train_loader,
        optimizer=args.optimizer,
        val_loader=val_loader,
        config_path=args.config,
    )
    print("Starting training...")
    trainer.train()