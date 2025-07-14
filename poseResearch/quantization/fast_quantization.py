from typing import Any, Dict, List, Optional
import warnings
import numpy
import torch
from transformers import AutoProcessor
from poseResearch.quantization.base_quantizer import VQVAEBase


class FASTQuantizer(VQVAEBase):
    def __init__(self, pretrained: Optional[str]=None) -> None:
        self.tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        if pretrained:
            self.tokenizer.from_pretrained(pretrained)
        else:
            self.tokenizer.bpe_tokenizer.add_special_tokens({"eos_token": "<EOS>"})
        

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.min_vals = torch.amin(x, dim=(1, 2), keepdim=True)
        self.max_vals = torch.amax(x, dim=(1, 2), keepdim=True)
        dist = self.max_vals - self.min_vals
        dist[dist == 0] = 1e-5  # Avoid division by zero
        return (x - self.min_vals) / dist

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.max_vals - self.min_vals) + self.min_vals

    def fit_tokenizer(self, data: torch.Tensor) -> None:
        normalized_tensor: numpy.ndarray = self._preprocess_input(data)
        # Replace NaN values with zeros and create block list
        normalized_tensor = numpy.nan_to_num(normalized_tensor, nan=0.0)
        block_list = [
            normalized_tensor[i]
            for i in range(normalized_tensor.shape[0])
        ]

        self.tokenizer.fit(block_list)

    def _preprocess_input(
        self, input_tensor: torch.Tensor, action_len: int = 8
    ) -> numpy.ndarray:
        """
        Preprocesses the input tensor to match the expected input format of the tokenizer.
        """
        input_tensor = input_tensor.squeeze()
        if len(input_tensor.size()) == 3:
            # Permute and flatten the input tensor
            inputs: torch.Tensor = input_tensor.permute(0, 2, 1).flatten(start_dim=1)
            assert (
                inputs.size(-1) == 17 * 3
            ), f"Expected flattended input to have {17*3} action dimension, got {inputs.size()}"
            pad_size = (action_len - inputs.size(0) % action_len) % action_len
            if pad_size > 0:
                inputs = torch.nn.functional.pad(inputs, (0, 0, 0, pad_size))
            # Reshape into chunks of action_len
            inputs = inputs.view(-1, action_len, inputs.size(-1))
            normalized_tensor = self.normalize(inputs)
            return normalized_tensor.numpy()
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
        return {
            "recovered": reshaped_decoded,
            "encoded": quantized,
            "loss": torch.mean(preprocessed - reshaped_decoded),  # counter padding
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
    
    def shape_back(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the input_ids back to the original shape.
        """
        # Assuming input_ids is a list of lists, where each inner list is a sequence of tokens
        if isinstance(input_ids, numpy.ndarray):
            input_ids = torch.tensor(input_ids)
        input_ids = self.denormalize(input_ids).unsqueeze(0)
        reshaped = input_ids.view(-1, 3, 17)
        reshaped = reshaped.permute(0, 2, 1)
        return reshaped  # Add batch dimension back

    def decode(self, input_ids: List[List[int]]) -> torch.Tensor:
        """
        Decodes the input_ids back to the original text.
        """
        decoded = self.tokenizer.decode(input_ids)
        return decoded

    def train_step(self, batch: torch.Tensor, optimizer, scheduler: Any = None):
        warnings.warn("Not implemented")
        return {"None": 0.0}

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    def load_parameters(self, path: str, strict: bool = True):
        self.tokenizer.from_pretrained(path)

    def save_tokenizer(self, path: str):
        self.tokenizer.save_pretrained(path)
