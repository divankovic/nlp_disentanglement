import torch.nn as nn
from utils.nn_utils import Flatten3D
import torch.nn.functional as F
import torch


class ConvEncoder(nn.Module):
    # The simplest case from paper - CNNs for sentence classification
    def __init__(self, input_dim, embedding_dim, latent_dim, args):
        # the total input will be something similar to Nx50x300x1  (stacked 50 words, 300dim embeddings for each)
        # input_dim - number of words
        # embedding_dim - the embedding_dimension
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.kernel_sizes = [3, 4, 5]
        self.num_kernels = 32
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_kernels, (kernel_size, embedding_dim)) for kernel_size in self.kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.hidden = nn.Linear(len(self.kernel_sizes) * self.num_kernels, 2*self.latent_dim)
        # in_channels, out_channels, kernel_size, stride, padding

    def forward(self, x):
        # x is of dim N x W x D - W- number of words, D - embedding dimension
        x = x.unsqueeze(1)  # Nx1xWxD
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convss]  # [(N, num_kernels, W), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, num_kernels), ...]*len(kernel_sizes)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(kernel_sizes)*num_kernels)
        z = self.hidden(x)  # (N, 2*latent_dim)
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
                  nn.Linear(256, latent_dim * 2)
                  )
}
