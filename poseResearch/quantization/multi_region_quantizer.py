import math
from typing import Any

import torch
from poseResearch.quantization.base_quantizer import VQVAEBase


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]  # type: ignore
        return self.dropout(x)


class TokenClassifier(torch.nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 256):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),  # reduce over L
        )
        self.classifier = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        # x: B × S × C × L
        B, S, C, L = x.shape
        x = x.view(B * S, C, L)  # Flatten sequences into batch
        x = self.encoder(x)  # (B*S, 64, 1)
        x = x.squeeze(-1)  # (B*S, 64)
        out = torch.argmax(self.classifier(x), -1)  # (B*S, num_classes)
        return out.unsqueeze(-1)  # (B, S, 1)


class TokenDecoder(torch.nn.Module):
    def __init__(
        self, num_classes: int = 256, out_channels: int = 3, token_length: int = 17
    ):
        super().__init__()
        self.codebook = torch.nn.Embedding(num_classes, out_channels * token_length)
        self.out_channels = out_channels
        self.token_length = token_length

    def forward(self, indices):
        # indices: B × S
        x = self.codebook(indices)  # (B, S, C*L)
        return x.view(
            x.size(0), x.size(1), self.out_channels, self.token_length
        )  # (B, S, C, L)


class Block(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.query = torch.nn.Linear(in_dim, in_dim)
        self.key = torch.nn.Linear(in_dim, in_dim)
        self.value = torch.nn.Linear(in_dim, in_dim)
        self.attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = torch.nn.LayerNorm(in_dim)
        self.norm2 = torch.nn.LayerNorm(out_dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_output, _ = self.attn(q, k, v)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        return self.norm2(x + ff_output)


class MultiRegionQuantizer(VQVAEBase):
    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__()
        self.nbooks = kwds.get("nbooks", 64)
        self.nregions = kwds.get("nregions", 4)
        self.emb_dim = kwds.get("n_e", 512)
        self.sequence_length = kwds.get(
            "sequence_length", 8
        )  # How many frames are considered jointly
        self.num_heads = kwds.get("num_heads", 4)
        self.dim_bottleneck = kwds.get("dim_bottleneck", 64)

        self.region_size = self.dim_bottleneck // self.nregions
        self.encoder = torch.nn.Sequential(
            # Perhaps without token classifier and embedding. Semantic closeness is given anyway.
            TokenClassifier(in_channels=3, num_classes=self.nbooks * self.region_size),
            torch.nn.Embedding(self.nbooks * self.region_size, self.emb_dim),
            PositionalEncoding(self.emb_dim, dropout=0.1, max_len=self.sequence_length),
            Block(self.emb_dim, self.emb_dim // self.nregions, self.num_heads),
            Block(self.emb_dim // self.nregions, self.dim_bottleneck, self.num_heads),
        )
        self.decoder = torch.nn.Sequential(
            PositionalEncoding(
                self.dim_bottleneck, dropout=0.1, max_len=self.sequence_length
            ),
            Block(self.dim_bottleneck, self.dim_bottleneck * 4, self.num_heads),
            Block(self.dim_bottleneck * 4, self.emb_dim, self.num_heads),
            TokenDecoder(
                num_classes=self.nbooks * self.region_size,
                out_channels=3,
                token_length=self.sequence_length,
            ),
        )

        self.codebooks = torch.Tensor((self.nregions, self.nbooks, self.region_size))

    def initialize_fresh(self, device: torch.device = torch.device("cpu")) -> None:
        """
        Initializes model parameters from scratch and moves the model to the given device.
        """

        # Initialize all submodules with default initialization
        def init_weights(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.apply(init_weights)

        # Initialize codebooks with random values on correct device
        self.codebooks = torch.nn.Parameter(
            torch.randn(self.nregions, self.nbooks, self.region_size),
            requires_grad=True,
        )

        # Move entire model to the specified device
        self.to(device)

    def load_parameters(self, path: str, strict: bool = True) -> None:
        """
        Loads model parameters from a checkpoint.

        Args:
            path (str): Path to the checkpoint.
            strict (bool): Whether to strictly enforce that the keys match.
        """
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if "codebooks" in checkpoint:
            self.codebooks = checkpoint["codebooks"].to(self.codebooks.device)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Forward pass through the model."""
        # x shape: (batch_size, sequence_length, emb_dim)
        batch_size, seq_len, _ = x.shape
        # Encode the input
        encoded = self.encoder(x)  # (batch_size, sequence_length, emb_dim)

        # Split encoded tensor into regions
        regions = encoded.chunk(
            self.nregions, dim=-1
        )  # List of (batch_size, sequence_length, region_size)

        # For each region, find the closest embedding from corresponding codebook
        quantized_regions = []
        for region, codebook in zip(regions, self.codebooks):
            # Calculate distances between region vectors and codebook embeddings
            distances = torch.cosine_similarity(
                region.reshape(-1, self.region_size), codebook
            )
            # Find closest codebook entries
            min_indices = distances.argmin(dim=1)
            # Get the corresponding embeddings
            quantized = codebook[min_indices].reshape(batch_size, seq_len, -1)
            quantized_regions.append(quantized)

        # Concatenate quantized regions back together
        encoded = torch.cat(
            quantized_regions, dim=-1
        )  # (batch_size, sequence_length, dim_bottleneck)

        # Decode the encoded regions
        decoded = self.decoder(
            encoded
        )  # (batch_size, sequence_length, out_channels, points)

        # Reshape to match the original input shape
        return {
            "recon": decoded,
            "loss": torch.tensor(
                0.0
            ),  # Placeholder for loss, to be computed in train_step
            "z_e": encoded,  # Encoded regions
        }

    def train_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
    ) -> dict[str, float]:
        """
        Performs a single training step.

        Args:
            batch (torch.Tensor): A batch of training data.
            optimizer (Optimizer): Optimizer instance.
            scheduler (Optional): Learning rate scheduler.

        Returns:
            Dict[str, float]: Logging info like total loss, VQ loss, recon loss, etc.
        """
        self.train()
        optimizer.zero_grad()

        output = self.forward(batch)
        recon_loss = torch.nn.functional.mse_loss(output["recon"], batch)

        # Backpropagation
        recon_loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        return {
            "loss": recon_loss.item(),
            "recon_loss": recon_loss.item(),
            "z_e": output["z_e"],
        }
