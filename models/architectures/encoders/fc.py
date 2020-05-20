import torch.nn as nn


class FCEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.main = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU()
        )

        self.mu = nn.Linear(100, latent_dim)
        self.logvar = nn.Linear(100, latent_dim)

    def forward(self, x):
        h = self.main(x.view(-1, self.input_dim))
        return self.mu(h), self.logvar(h)
