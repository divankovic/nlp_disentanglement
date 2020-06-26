import torch.nn as nn
from torch import sigmoid
import probtorch
import torch
from utils.torch_utils import ScaledSoftmax


class FCDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, architecture='basic', **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.main = ARCHITECTURES[architecture](latent_dim, output_dim)

    def forward(self, z):
        # return sigmoid(self.main(z))
        return self.main(z)


class PTFCDecoder(nn.Module):

    def __init__(self, latent_dim, output_dim, batch_size, architecture='NTM', **kwargs):
        super().__init__()
        self.main = ARCHITECTURES[architecture](latent_dim, output_dim)
        self.architecture = architecture
        if architecture == 'GSM':
            torch.nn.init.uniform_(self.main[1].weight, a=0.0, b=10.0)  # will init to (0,1)
        elif architecture == 'GSM_scale':
            print('Using GSM_scale architecture!')
        self.prior_mean = torch.zeros((batch_size, latent_dim)).cuda().double()
        self.prior_cov = torch.eye(latent_dim).cuda().double()

    def forward(self, x, q):
        p = probtorch.Trace()
        z = p.multivariate_normal(self.prior_mean, self.prior_cov, value=q['z'], name='z')
        if self.architecture == 'GSM_BN':
            num_samples = z.shape[0]
            z = z[0, :, :]
        x_recon = self.main(z)
        if self.architecture == 'GSM_BN':
            x_recon = x_recon.expand(num_samples, *x_recon.size())
        p.loss(lambda x_recon, x: -(torch.log(x_recon) * x).sum(-1),
               x_recon, x, name='x_recon')
        return p


class HFCDecoder(PTFCDecoder):
    # for structured (hierarchical) 2d latent representations
    def __init__(self, latent_dim, output_dim, batch_size, num_groups, architecture='NTM', **kwargs):
        super().__init__(latent_dim, output_dim, batch_size, architecture, **kwargs)
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
        if self.architecture == 'GSM_BN':
            latents = latents[0, :, :]
            num_samples = z.shape[0]
        x_recon = self.main(latents)
        if self.architecture == 'GSM_BN':
            x_recon = x_recon.expand(num_samples, *x_recon.size())
        p.loss(lambda x_recon, x: -(torch.log(x_recon) * x).sum(-1), x_recon, x, name='x_recon')
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
        nn.Linear(latent_dim, output_dim, bias=False),
        nn.Softmax(dim=-1)
    ),
    'NTM': lambda latent_dim, output_dim:
    nn.Sequential(
        nn.ReLU(),
        nn.Linear(latent_dim, output_dim, bias=False),
        nn.Softmax(dim=-1)
    ),
    'GSM': lambda latent_dim, output_dim:
    nn.Sequential(
        nn.Softmax(dim=-1),
        nn.Linear(latent_dim, output_dim, bias=False),
        nn.Softmax(dim=-1)
    ),
    'GSM_scale': lambda latent_dim, output_dim:
    nn.Sequential(
        ScaledSoftmax(100),
        nn.Linear(latent_dim, output_dim, bias=False),
        nn.Softmax(dim=-1)
    ),
    'GSM_BN': lambda latent_dim, output_dim:
    nn.Sequential(
        nn.Softmax(dim=-1),
        nn.BatchNorm1d(latent_dim),
        nn.Linear(latent_dim, output_dim, bias=False),
        nn.BatchNorm1d(output_dim),
        nn.Softmax(dim=-1)
    )

}
