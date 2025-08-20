from typing import Any, Dict, List, Optional
import numpy as np
import torch
from transformers import AutoProcessor
from poseResearch.quantization.base_quantizer import VQVAEBase
import json
import os
import logging
import sys


# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Default: log everything; filter with handlers

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # Default: show INFO and above

# Format including file name
formatter = logging.Formatter('%(levelname)s | %(filename)s:%(lineno)d | %(message)s')
console_handler.setFormatter(formatter)

# Add the handler
logger.addHandler(console_handler)


class FASTQuantizer(VQVAEBase):
    def __init__(self, pretrained: Optional[str] = None) -> None:
        self.tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        if pretrained is not None:
            if not os.path.exists(pretrained):
                raise ValueError(f"Pretrained path {pretrained} does not exist")
            with open(f"{pretrained}/processor_config.json", "r") as f:
                config = json.load(f)
            self.tokenizer = self.tokenizer.from_pretrained(pretrained, trust_remote_code=True, vocab_size=config.get("vocab_size", 1024))
        else:
            self.tokenizer.bpe_tokenizer.add_special_tokens({"eos_token": "<EOS>"})
        self.next_joint_first = False

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.min_vals = torch.amin(x, dim=(1, 2), keepdim=True)
        self.max_vals = torch.amax(x, dim=(1, 2), keepdim=True)
        dist = self.max_vals - self.min_vals
        dist[dist == 0] = 1e-5  # Avoid division by zero
        return ((x - self.min_vals) / dist) * 10.

    def denormalize(self, x: torch.Tensor, i: Optional[int] = None) -> torch.Tensor:
        if i is not None and i < len(self.min_vals):
            return x * (self.max_vals[i][0] - self.min_vals[i][0])/10. + self.min_vals[i][0]
        return x * (self.max_vals[x.size(0)][0] - self.min_vals[x.size(0)][0]) + self.min_vals[x.size(0)][0]

    def fit_tokenizer(self, data: torch.Tensor, next_joint_first: bool = False, num_tokens: int = -1) -> None:
        self.next_joint_first = next_joint_first
        normalized_tensor: np.ndarray = self._preprocess_input(data).numpy()
        # Replace NaN values with zeros and create block list
        normalized_tensor = np.nan_to_num(normalized_tensor, nan=0.0)
        block_list = [normalized_tensor[i] for i in range(normalized_tensor.shape[0])]
        self.tokenizer.vocab_size = num_tokens if num_tokens > 0 else 1024
        self.tokenizer = self.tokenizer.fit(block_list, vocab_size=num_tokens if num_tokens > 0 else 1024)
        self.tokenizer.bpe_tokenizer.add_special_tokens({"eos_token": "<EOS>"})

    def _preprocess_input(
        self, input_tensor: torch.Tensor, action_len: int = 8
    ) -> torch.Tensor:
        """
        Preprocesses the input tensor to match the expected input format of the tokenizer.
        """
        input_tensor = input_tensor.squeeze()
        if len(input_tensor.size()) == 3:
            # Permute and flatten the input tensor
            if not self.next_joint_first:
                input_tensor = input_tensor.permute(0, 2, 1)
            inputs: torch.Tensor = input_tensor.flatten(start_dim=1)
            assert (
                inputs.size(-1) == 17 * 3
            ), f"Expected flattended input to have {17*3} action dimension, got {inputs.size()}"
            pad_size = (action_len - inputs.size(0) % action_len) % action_len
            if pad_size > 0:
                inputs = torch.nn.functional.pad(inputs, (0, 0, 0, pad_size))
            # Reshape into chunks of action_len
            inputs = inputs.view(-1, action_len, inputs.size(-1))
            normalized_tensor = self.normalize(inputs)
            return normalized_tensor
        else:
            raise ValueError(
                f"Expected input tensor to have 3 dimensions (B, 17, 3), got {input_tensor.size()}. Please check the input format."
            )

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        preprocessed = self._preprocess_input(x)
        quantized = self.quantize(preprocessed)
        
        decoded = self.decode(quantized)
        decoded = torch.tensor(decoded, dtype=torch.float32)
        reshaped_decoded = decoded.view(-1, 3, 17)  # Reshape back to (B, 3, 17)
        reshaped_decoded = reshaped_decoded.permute(0, 2, 1)  # Permute to (B, 17, 3)
        preprocessed = preprocessed.view(-1, 3, 17).permute(0, 2, 1)
        
        root_to_spine_distances = []
        for pose in preprocessed:
            root_point = pose[0]
            spine_point = pose[7]

            # Calculate Euclidean distance between root and spine
            distance = np.linalg.norm(spine_point - root_point)
            if distance > 0:  # Avoid division by zero
                root_to_spine_distances.append(distance)

        if not root_to_spine_distances:
            scale_factor = 1.0  # Default scale factor if no valid distances

        # Use median distance to avoid outliers
        median_measured_distance = np.median(root_to_spine_distances)

        # Calculate scale factor: real_distance / measured_distance
        scale_factor = (
            17.*16. / median_measured_distance
        )
            ###

        diff = preprocessed - reshaped_decoded
        dists = torch.norm(diff, dim=1)  # Euclidean norm over xyz → (B, K, T)

            # MPJPE per pose (mean over joints K)
        mpjpe_per_pose = dists.mean(dim=1)  # → (B, T)

            # Final MPJPE (mean over all poses)
        mpjpe = mpjpe_per_pose.mean().item()

        return {
            "recovered": reshaped_decoded,
            "encoded": quantized,
            "loss": torch.mean(preprocessed - reshaped_decoded),
            "mpjpe": mpjpe * scale_factor,  # counter padding
        }

    def quantize(self, input_tensor: torch.Tensor) -> List[List[int]]:
        """
        Quantizes the input text using the fitted tokenizer.
        """
        if input_tensor.size(-1) == 17 * 3:
            inputs = input_tensor.numpy()
        else:
            inputs = self._preprocess_input(input_tensor)

        out = self.tokenizer(inputs)

        return out

    def shape_back(self, input_ids: torch.Tensor, i: Optional[int] = None) -> torch.Tensor:
        """
        Reshapes the input_ids back to the original shape.
        """
        # Assuming input_ids is a list of lists, where each inner list is a sequence of tokens
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.tensor(input_ids)
        input_ids = self.denormalize(input_ids, i).unsqueeze(0)
        reshaped = input_ids.view(-1, 3, 17)
        reshaped = reshaped.permute(0, 2, 1)
        return reshaped  # Add batch dimension back

    def decode(self, input_ids: List[List[int]]) -> Optional[torch.Tensor]:
        """
        Decodes the input_ids back to the original text.
        """
        max_retries = 5
        for retry in range(max_retries):
            decoded = self.tokenizer.decode(input_ids)
            if (decoded==0.).all():
                if retry == max_retries - 1:
                    logger.error("Terminated decoding without success")
                    return None
                input_ids[-1].append(0)  # Append last token to avoid empty decoding
            else:
                logger.info("Decoded successfully")
                return decoded

    def train_step(self, batch: torch.Tensor, optimizer, scheduler: Any = None):
        logger.error("Not implemented")
        return {"None": 0.0}

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    def load_parameters(self, path: str, strict: bool = True):
        self.tokenizer.from_pretrained(path)

    def save_tokenizer(self, path: str):
        self.tokenizer.save_pretrained(path)
