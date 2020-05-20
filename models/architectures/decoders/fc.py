import torch.nn as nn
from torch import sigmoid


class FCDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 400),
            nn.ReLU(),
            nn.Linear(400, output_dim)
        )

    def forward(self, z):
        return sigmoid(self.main(z))
