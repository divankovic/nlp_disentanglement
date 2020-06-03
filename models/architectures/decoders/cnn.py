import torch.nn as nn
from utils.nn_utils import Unsqueeze3D, View
from torch import sigmoid


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, embedding_dim):
        # the output input will be something similar to Nx50x300x1  (stacked 50 words, 300dim embeddings for each)
        # output_dim - number of words
        # embedding_dim - the embedding_dimension
        # reverse form simple conv encoder
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.kernel_sizes = [3, 4, 5]
        self.num_kernels = 32

        self.latent_to_hidden = nn.Linear(self.latent_dim, len(self.kernel_sizes) * self.num_kernels)
        self.deconvs = nn.ModuleList(
            [nn.ConvTranspose2d(self.num_kernels, 1, (kernel_size, embedding_dim)) for kernel_size in
             self.kernel_sizes])
        # TODO - finish this up

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
