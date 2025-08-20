import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(
        self,
        h_dim,
        res_h_dim,
        n_res_layers,
        n_embeddings,
        embedding_dim,
        beta,
        save_img_embedding_map=False,
    ):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim // 10, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def tokenize(self, x, pad=3):
        torch.nn.functional.pad(x, (0, 0, 0, pad), "constant", 0)

        z_e = self.encoder(x)

        embedding_loss, z_q, perplexity, _, token = self.vector_quantization(z_e)
        return token[0][0]

    def detokenize(self, token):
        # get quantized latent vectors
        z_q = self.vector_quantization.embedding.weight[token]
        z_q = z_q.view(-1, 4, 5, 2)
        x_hat = self.decoder(z_q)
        return x_hat

    def forward(self, x, verbose=False, pad: int = 3):

        x = torch.nn.functional.pad(x, (0, 0, 0, pad), "constant", 0)

        z_e = self.encoder(x)
        if verbose:
            print("z_e shape:", z_e.size())

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print("original data shape:", x.shape)
            print("encoded data shape:", z_e.shape)
            print("recon data shape:", x_hat.shape)
            print("pad:", pad)
            assert False

        return embedding_loss, x_hat[:, :, :-pad, :], perplexity
