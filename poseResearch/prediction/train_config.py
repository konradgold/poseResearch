import torch
import yaml
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # wandb settings
    wandb_log: bool = False
    wandb_project: str = "pose_research"
    wandb_run_name: str = "gpt2-train"

    # training settings
    batch_size: int = 12
    block_size: int = 128
    gradient_accumulation_steps: int = 5

    # model settings
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False

    # optimization settings
    max_iters: int = 600000
    lr_decay_iters: int = 600000
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    vocab_size: int = 2048

    # learning rate decay settings
    decay_lr: bool = True
    warmup_iters: int = 2000
    min_lr: float = 6e-5

    # evaluation settings
    eval_interval: int = 1000
    eval_iters: int = 200
    log_interval: int = 10

    # system settings
    out_dir: str = "out"
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"
    dataset: str = "overfit"
    backend: str = "nccl"
    compile: bool = False

    backend = "nccl"  # 'nccl', 'gloo', etc.
    # system
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  #

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load config from yaml file, using default values for missing fields."""
        try:
            with open(yaml_path, "r") as f:
                yaml_config = yaml.safe_load(f)
            # Create default config
            config = cls()
            # Update with yaml values
            if yaml_config is not None:  # Handle empty yaml file
                for key, value in yaml_config.items():
                    if hasattr(config, key):
                        # Convert value to the same type as the default value
                        default_value = getattr(config, key)
                        converted_value = type(default_value)(value)
                        setattr(config, key, converted_value)
            return config
        except FileNotFoundError:
            print(f"Config file {yaml_path} not found. Using default values.")
            return cls()
        except yaml.YAMLError as e:
            print(f"Error parsing yaml file: {e}")
            return cls()

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save_yaml(self, yaml_path: str) -> None:
        """Save config to yaml file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f)
