import torch.nn as nn
import torch
import probtorch
from probtorch.util import expand_inputs


class FCEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, architecture='basic', **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.architecture = architecture
        self.main = ARCHITECTURES[architecture](input_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.main(x)
        return self.mu(h), self.logvar(h)


class PTFCEncoder(FCEncoder):
    # tuned to work with probtorch
    def __init__(self, input_dim, hidden_dim, latent_dim, architecture='NTM', **kwargs):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.main = ARCHITECTURES[architecture](input_dim, hidden_dim)

    @expand_inputs
    def forward(self, x, num_samples=1):
        if num_samples is not None:
            x = x.expand(num_samples, *x.size())
        q = probtorch.Trace()
        h = self.main(x)
        q.normal(self.mu(h), torch.exp(self.logvar(h) * 0.5), name='z')

        return q


class HFCEncoder(PTFCEncoder):
    # for structured (hierarchical) 2d latent representations
    def __init__(self, input_dim, hidden_dim, latent_dim, num_groups, architecture='NTM', **kwargs):
        super().__init__(input_dim, hidden_dim, latent_dim, architecture, **kwargs)
        if latent_dim % num_groups != 0:
            raise ValueError('Latent_dim must be disible by num_groups!')
        self.num_groups = num_groups
        self.group_dim = int(self.latent_dim/self.num_groups)

    @expand_inputs
    def forward(self, x, num_samples=1):
        if num_samples is not None:
            x = x.expand(num_samples, *x.size())
        q = probtorch.Trace()
        h = self.main(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        for i in range(self.num_groups):
            q.normal(mu[..., i * self.group_dim: (i + 1) * self.group_dim],
                     torch.exp(logvar[..., i * self.group_dim: (i + 1) * self.group_dim] * 0.5), name='z_' + str(i))

        return q


ARCHITECTURES = {
    'basic': lambda input_dim, hidden_dim:
    nn.Sequential(
        nn.Linear(input_dim, 400),
        nn.ReLU(),
        nn.Linear(400, hidden_dim),
        nn.ReLU()
    ),
    'NVDM': lambda input_dim, hidden_dim:
    nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU()
    ),
    'NTM': lambda input_dim, hidden_dim:
    nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU()
    ),
    'NVDM+': lambda input_dim, hidden_dim:
    nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU()
    )
}
