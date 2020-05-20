import torch.nn as nn
from utils.nn_utils import Unsqueeze3D, View
from torch import sigmoid


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, nc=1):
        # nc = number of channels, is 1 for textual data usually
        # assert len(input_dim) == 2, 'A matrix for text representation (ex. stacked word embeddings)'
        # assume input_dim similar to 50*300  (stacked 50 words, 300dim embeddings for each)
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.main = cnn_architectures['simpleconv'](nc, latent_dim)

    def forward(self, x):
        return sigmoid(self.main(x))


cnn_architectures = {
    'simpleconv': lambda nc, latent_dim:
    nn.Sequential(
        # mnist example - modify for textual data
        nn.Linear(latent_dim, 256),  # B, 256
        Unsqueeze3D(),  # B, 256,  1,  1
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
        nn.ReLU(True),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
        nn.ReLU(True),
        nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
    ),
    'conv+': lambda nc, latent_dim:
    nn.Sequential(
        nn.Linear(latent_dim, 256),  # B, 256
        nn.ReLU(True),
        nn.Linear(256, 256),  # B, 256
        nn.ReLU(True),
        nn.Linear(256, 32 * 4 * 4),  # B, 512
        nn.ReLU(True),
        View((-1, 32, 4, 4)),  # B,  32,  4,  4
        nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
        nn.ReLU(True),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
        nn.ReLU(True),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
        nn.ReLU(True),
        nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B,  nc, 64, 64
    )
}
