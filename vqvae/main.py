import numpy as np
import torch
import torch.optim as optim
import argparse
import utils as utils
from models.vqvae import VQVAE
from MotionBERT.lib.model.loss import mpjpe, p_mpjpe

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=500)
parser.add_argument("--n_hiddens", type=int, default=-1)
parser.add_argument("--n_residual_hiddens", type=int, default=2)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=80)  # should be 10*hiddens
parser.add_argument("--n_embeddings", type=int, default=2000)
parser.add_argument("--beta", type=float, default=0.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset", type=str, default="keypoints")

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename", type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print("Results will be saved in ./results/vqvae_" + args.filename + ".pth")

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, x_train_var = (
    utils.load_data_and_data_loaders(args.dataset, args.batch_size)
)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""


def iterate():
    mpjpe = []
    for hidden in range(4, 40):
        if args.n_hiddens == -1:
            n_hiddens = hidden
        else:
            n_hiddens = args.n_hiddens
        embedding_dim = 10 * n_hiddens
        model = VQVAE(
            n_hiddens,
            args.n_residual_hiddens,
            args.n_residual_layers,
            args.n_embeddings,
            embedding_dim,
            args.beta,
        ).to(device)

        """
        Set up optimizer and training loop
        """
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

        model.train()

        results = {
            "n_updates": 0,
            "recon_errors": [],
            "loss_vals": [],
            "perplexities": [],
            "mpjpe": [],
        }

        def train():

            for i in range(args.n_updates):
                x = next(iter(training_loader))
                x = x[0].to(device)
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
            x = x[0].to(device)
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

            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, results, hyperparameters, args.filename
                )

            print(
                "Result for hidden#",
                hidden,
                "Recon Error:",
                np.mean(results["recon_errors"][-args.log_interval :]),
                "Loss",
                np.mean(results["loss_vals"][-args.log_interval :]),
                "Perplexity:",
                np.mean(results["perplexities"][-args.log_interval :]),
                "MPJPE:",
                np.mean(results["mpjpe"]),
            )
            return model

        mpjpe.append(train())
        if args.n_hiddens > 0:
            break
    print(mpjpe)


if __name__ == "__main__":
    iterate()
