import torch.nn as nn
import torch
import probtorch
from probtorch.util import expand_inputs


class FCEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.main = ARCHITECTURES['basic'](input_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.main(x.view(-1, self.input_dim))
        return self.mu(h), self.logvar(h)


class HFVAEFCEncoder(FCEncoder):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(HFVAEFCEncoder, self).__init__(input_dim, hidden_dim, latent_dim)
        self.main = ARCHITECTURES['HFVAE_NVDM'](input_dim, hidden_dim)

    @expand_inputs
    def forward(self, x, num_samples=1):
        if num_samples is not None:
            x = x.expand(num_samples, *x.size())
        q = probtorch.Trace()
        h = self.main(x)
        q.normal(self.mu(h), torch.exp(self.logvar(h) * 0.5), name='z')

        return q


ARCHITECTURES = {
    'basic': lambda input_dim, hidden_dim:
    nn.Sequential(
        nn.Linear(input_dim, 400),
        nn.ReLU(),
        nn.Linear(400, hidden_dim),
        nn.ReLU()
    ),
    'HFVAE_NVDM': lambda input_dim, hidden_dim:
    nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU()
    )
}
