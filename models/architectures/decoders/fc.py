import torch.nn as nn
from torch import sigmoid
import probtorch
import torch


class FCDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, architecture='basic'):
        super().__init__()
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.main = ARCHITECTURES[architecture](latent_dim, output_dim)

    def forward(self, z):
        # return sigmoid(self.main(z))
        return self.main(z)


class PTFCDecoder(nn.Module):

    def __init__(self, latent_dim, output_dim, batch_size):
        super().__init__()
        self.main = ARCHITECTURES['NVDM'](latent_dim, output_dim)
        self.prior_mean = torch.zeros((batch_size, latent_dim)).cuda().double()
        self.prior_cov = torch.eye(latent_dim).cuda().double()

    def forward(self, x, q):
        p = probtorch.Trace()
        z = p.multivariate_normal(self.prior_mean, self.prior_cov, value=q['z'], name='z')
        x_recon = self.main(z)
        p.loss(lambda x_recon, x: -(torch.log(x_recon + 1e-8) * x).sum(-1),
               x_recon, x, name='x_recon')
        return p


class HFCDecoder(PTFCDecoder):
    # for structured (hierarchical) 2d latent representations
    def __init__(self, latent_dim, output_dim, batch_size, num_groups):
        super().__init__(latent_dim, output_dim, batch_size)
        if latent_dim % num_groups != 0:
            raise ValueError('Latent_dim must be disible by num_groups!')
        self.num_groups = num_groups
        self.group_dim = int(latent_dim / num_groups)
        self.prior_mean = torch.zeros((batch_size, self.group_dim)).cuda().double()
        self.prior_cov = torch.eye(self.group_dim).cuda().double()

    def forward(self, x, q):
        p = probtorch.Trace()
        zs = []
        for i in range(self.num_groups):
            z = p.multivariate_normal(loc=self.prior_mean, covariance_matrix=self.prior_cov, value=q['z_' + str(i)],
                                      name='z_' + str(i))
            zs.append(z)

        latents = torch.cat(zs, -1)
        x_recon = self.main(latents)
        p.loss(lambda x_recon, x: -(torch.log(x_recon+1e-8)*x).sum(-1), x_recon, x, name='x_recon')
        return p


ARCHITECTURES = {
    'basic': lambda latent_dim, output_dim:
    nn.Sequential(
        nn.Linear(latent_dim, 100),
        nn.ReLU(),
        nn.Linear(100, 400),
        nn.ReLU(),
        nn.Linear(400, output_dim)
    ),
    'NVDM': lambda latent_dim, output_dim:
    nn.Sequential(
        nn.Linear(latent_dim, output_dim),
        nn.Softmax(dim=-1)
    )
}
