import argparse
import json
from logging import warning
from typing import List
import numpy as np
import torch
from pathlib import Path
from models.vqvae import VQVAE
import torch.optim as optim
import utils



def load_json(path: str) -> List:
    """Load JSON data from file."""
    file_path = Path(path)
    if not file_path.is_dir():
        raise ValueError(f"{file_path} is not a directory")

    json_files = list(file_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {file_path}")

    data_list = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data_list.append(json.load(f))

    return data_list


def process_data(data: List) -> torch.Tensor:
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
        # Your processing logic here
        data[i] = poses
    prepared_data = torch.cat(data, dim=0)
    prepared_data.unsqueeze(0)

    return prepared_data


def save_tensor(tensor: torch.Tensor, output_path: str) -> None:  # type: ignore
    """Save tensor to binary file."""
    output_path: Path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_path)


def train(model, args, training_loader, validation_loader, optimizer, x_train_var):
    results = {
        "recon_errors": [],
        "perplexities": [],
        "loss_vals": [],
        "mpjpe": [],
        "n_updates": 0,
    }
    for i in range(args.n_updates):
        x = next(iter(training_loader))
        x = x[0]
        x = x.permute(0, 3, 2, 1)

        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x, pad=3)
        recon_loss = torch.mean((x_hat - x) ** 2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

    x = next(iter(validation_loader))
    x = x[0]
    x = x.permute(0, 3, 2, 1)
    optimizer.zero_grad()
    with torch.no_grad():
        embedding_loss, x_hat, perplexity = model(x, pad=3)
    recon_loss = torch.mean((x_hat - x) ** 2) / x_train_var
    loss = recon_loss + embedding_loss

    results["recon_errors"].append(recon_loss.cpu().detach().numpy())
    results["perplexities"].append(perplexity.cpu().detach().numpy())
    results["loss_vals"].append(loss.cpu().detach().numpy())

    ###
    root_to_spine_distances = []
    for pose in x.permute(0, 3, 2, 1).reshape(-1, 17, 3):
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
    scale_factor = 17.0 * 16.0 / median_measured_distance
    ###

    diff = x - x_hat
    dists = torch.norm(diff, dim=1)  # Euclidean norm over xyz → (B, K, T)

    # MPJPE per pose (mean over joints K)
    mpjpe_per_pose = dists.mean(dim=1)  # → (B, T)

    # Final MPJPE (mean over all poses)
    mpjpe = mpjpe_per_pose.mean().item()
    results["mpjpe"].append(mpjpe * scale_factor)

    return model


def main():
    parser = argparse.ArgumentParser(description="Process JSON file and save as tensor")
    parser.add_argument("--tokenizer", type=str, help="What tokenizer to use", default="poseResearch/prediction/data/tokenizer")
    parser.add_argument(
        "--input_path", type=str, help="Path to directory of json files", default="/Volumes/KG1TB/Developement/poseResearch/poseResearch/dataloader/male2_t2_cam01"
    )
    parser.add_argument("--filename", type=str, help="Path to output .bin file", default="/Volumes/KG1TB/Developement/poseResearch/data/fit11xvqvae")

    timestamp = utils.readable_timestamp()

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_updates", type=int, default=500)
    parser.add_argument("--n_hiddens", type=int, default=8)
    parser.add_argument("--n_residual_hiddens", type=int, default=2)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=80)  # should be 10*hiddens
    parser.add_argument("--n_embeddings", type=int, default=2000)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="keypoints")

    # whether or not to save model


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Load data and define batch data loaders
    """
    """
    Set up VQ-VAE model with components defined in ./models/ folder
    """


    args = parser.parse_args()

    model = VQVAE(
                args.n_hiddens,
                args.n_residual_hiddens,
                args.n_residual_layers,
                args.n_embeddings,
                args.embedding_dim,
                args.beta,
            ).to(device)
    
    training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(args.dataset, args.batch_size)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    model = train(model, args, training_loader, validation_loader, optimizer, x_train_var)

    model.eval()

    # Collect tokenized sequences for train and val
    tokenized_train = []
    tokenized_val = []

    # Tokenize training data
    for batch in training_loader:
        x = batch[0].to(device)
        tokens = model.tokenize(x, pad=3)  # expect shape (B, L)
        tokens = [item for sublist in tokens.cpu().numpy().tolist() for item in sublist]
        # append EOT token to each sequence
        tokenized_train += tokens

    print(tokenized_train)
    # Tokenize validation data
    for batch in validation_loader:
        x = batch[0].to(device)
        tokens = model.tokenize(x, pad=3)
        tokens = [item for sublist in tokens.cpu().numpy().tolist() for item in sublist]
        tokenized_val += tokens
    print(tokenized_val)

    # Save to memmap or .bin files
    train_out = f"{args.filename}/train.bin"
    val_out   = f"{args.filename}/val.bin"

    # Flatten and save
    flat_train = np.array(tokenized_train).astype(np.uint16)
    flat_val   = np.array(tokenized_val).astype(np.uint16)

    arr_train = np.memmap(train_out, dtype=np.uint16, mode="w+", shape=flat_train.shape)
    arr_train[:] = flat_train
    arr_train.flush()

    arr_val = np.memmap(val_out, dtype=np.uint16, mode="w+", shape=flat_val.shape)
    arr_val[:] = flat_val
    arr_val.flush()

    print(f"Saved train tokens → {train_out}")
    print(f"Saved val tokens   → {val_out}")


if __name__ == "__main__":
    main()
