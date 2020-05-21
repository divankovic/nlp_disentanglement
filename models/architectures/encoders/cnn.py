import torch.nn as nn
from utils.nn_utils import Flatten3D

# TODO - research how cnn's were used on text
class ConvEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, nc=1):
        # nc = number of channels, is 1 for textual data usually
        # assert len(input_dim) == 2, 'A matrix for text representation (ex. stacked word embeddings)'
        # assume input_dim similar to 50*300  (stacked 50 words, 300dim embeddings for each)
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.main = cnn_architectures['simpleconv'](nc, latent_dim)

    def forward(self, x):
        z = self.main(x)
        mu = z[:, :self.latent_dim]
        logvar = z[:, self.latent_dim:]
        return mu, logvar


cnn_architectures = {
    'simpleconv': lambda nc, latent_dim:
    nn.Sequential(
        # in_channels, out_channels, kernel_size, stride, padding
        nn.Conv2d(nc, 32, 4, 2, 1),
        nn.ReLU(True),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ReLU(True),
        nn.Conv2d(32, 64, 4, 2, 1),
        nn.ReLU(True),
        nn.Conv2d(64, 256, 4, 1),
        nn.ReLU(True),
        Flatten3D(),
        nn.Linear(256, latent_dim * 2)
    ),
    'conv+': lambda nc, latent_dim:
    nn.Sequential(nn.Conv2d(nc, 32, 4, 2, 1),
                  nn.ReLU(True),
                  nn.Conv2d(32, 32, 4, 2, 1),
                  nn.ReLU(True),
                  nn.Conv2d(32, 32, 4, 2, 1),
                  nn.ReLU(True),
                  nn.Conv2d(32, 32, 4, 2, 1),
                  nn.ReLU(True),
                  Flatten3D(),
                  nn.Linear(32 * 4 * 4, 256),
                  nn.ReLU(True),
                  nn.Linear(256, 256),
                  nn.ReLU(True),
                  nn.Linear(256, latent_dim * 2)
                  ),
    'padlessconv': lambda nc, latent_dim:
    nn.Sequential(nn.Conv2d(nc, 32, 3, 2, 0),
                  nn.ReLU(True),
                  nn.Conv2d(32, 32, 3, 2, 0),
                  nn.ReLU(True),
                  nn.Conv2d(32, 64, 3, 2, 0),
                  nn.ReLU(True),
                  nn.Conv2d(64, 128, 3, 2, 0),
                  nn.ReLU(True),
                  nn.Conv2d(128, 256, 3, 2, 0),
                  nn.ReLU(True),
                  Flatten3D(),
                  nn.Linear(256, latent_dim*2)
                  )
}
