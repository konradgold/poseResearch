import argparse
import json
from logging import warning
from typing import List
from networkx import set_node_attributes
from numpy import dtype
import numpy
import torch
from pathlib import Path
from poseResearch.quantization.fast_quantization import FASTQuantizer

def load_json(path: str) -> List:
    """Load JSON data from file."""
    file_path = Path(path)
    if not file_path.is_dir():
        raise ValueError(f"{file_path} is not a directory")
        
    json_files = list(file_path.glob('*.json'))
    if not json_files:
        raise ValueError(f"No JSON files found in {file_path}")

    data_list = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data_list.append(json.load(f))

    return data_list


def process_data(data: List, tokenizer: str, fit: bool) -> torch.Tensor:
    """
    Process the loaded JSON data and return a torch.Tensor of shape (N, M).
    
    Args:
        data: The loaded JSON data
        tokenizer: The tokenizer to use for processing

    Returns:
        torch.Tensor: Processed tensor of shape (N, M)
    """
    
    for i, poses in enumerate(data):
        poses = torch.Tensor(poses["poselifting"]["data"]).squeeze()
        print(f"Processing data for index {i}, shape: {poses.size()}")
        assert len(poses.size()) <= 4
        assert len(poses.size()) >= 3
        if len(poses.size()) == 4:
            warning("Not tested for multiple persons.")
            poses = poses.reshape(-1, *poses.shape[2:])
        assert len(poses.size()) == 3
        assert poses.size(-2) == 17
        assert poses.size(-1) == 3
        poses = poses.unsqueeze(0)  # Add batch dimension
        # Your processing logic here
        data[i] = poses
    prepared_data = torch.cat(data, dim=0)

    if fit:
        fast_tokenizer = FASTQuantizer()
        fast_tokenizer.fit_tokenizer(prepared_data)
        # Your tokenization logic here
        result_tensor = fast_tokenizer.quantize(prepared_data)
        fast_tokenizer.save_tokenizer(tokenizer)
    else:
        fast_tokenizer = FASTQuantizer(tokenizer)
        result_tensor = fast_tokenizer.quantize(prepared_data)
        # Your non-tokenization logic here
    


    print(f"Nr. Tokens: {fast_tokenizer.vocab_size}")
    return result_tensor


def save_tensor(tensor: torch.Tensor, output_path: str) -> None: # type: ignore
    """Save tensor to binary file."""
    output_path: Path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_path)


def main():
    parser = argparse.ArgumentParser(description='Process JSON file and save as tensor')
    parser.add_argument('--tokenizer', type=str, 
                       help='What tokenizer to use')
    parser.add_argument('--input_path', type=str, 
                       help='Path to directory of json files')
    parser.add_argument('--output_path', type=str, 
                       help='Path to output .bin file')
    parser.add_argument('--fit', action="store_true", help="Whether to fit the tokeniser")

    args = parser.parse_args()

    
    # Load JSON data
    print(f"Loading JSON from: {args.input_path}")
    data = load_json(args.input_path)
    
    # Process data
    print(f"Processing data (fit tokenizer={args.fit})...")
    result_tensor = process_data(data, args.tokenizer, fit=args.fit)
    
    # Save tensor

    result_tensor = numpy.concatenate([x for x in result_tensor])
    arr = numpy.memmap(args.output_path, dtype=numpy.uint16, mode='w+', shape=(len(result_tensor),))
    arr[:] = result_tensor[:]

    
    print("Processing complete!")


if __name__ == "__main__":
    main()