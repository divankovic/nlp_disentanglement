from models.vae import VAE
import probtorch
from torch import nn
import torch


# ProbTorch VAE implementations
# vae implemetation modified to be compatible with probtorch
class PTVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        q = self.encoder(x)
        p = self.decoder(x, q)
        return q, p

    def loss_function(self, q, p, reduce=True, **kwargs):
        return - probtorch.objectives.montecarlo.elbo(q, p, sample_dim=0, batch_dim=1, beta=1.0, reduce=reduce)


class HFVAE(PTVAE):

    def __init__(self, encoder, decoder, beta=None):
        super().__init__(encoder, decoder)
        if beta is None:
            # beta template for hfvae (gammma, 1, alpha, beta, 0)
            beta = (1.0, 1.0, 1.0, 1.0, 0)
        self.beta = beta

    def loss_function(self, q, p, reduce=True, **kwargs):
        N = kwargs['N']
        batch_size = kwargs['batch_size']
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.0
        bias = (N - 1) / (batch_size - 1)
        return -probtorch.objectives.marginal.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha, beta=self.beta,
                                                       bias=bias, reduce=reduce)

    def mutual_info(self, q, p, **kwargs):
        N = kwargs['N']
        batch_size = kwargs['batch_size']
        bias = (N - 1) / (batch_size - 1)
        sample_dim = 0
        batch_dim = 1
        z = [n for n in q.sampled() if n in p]
        log_qz = q.log_joint(sample_dim, batch_dim, z)
        log_joint_avg_qz, _, _ = q.log_batch_marginal(sample_dim, batch_dim, z, bias=bias)

        return (log_qz - log_joint_avg_qz).mean()

    def mutual_info_by_components(self, q, p, **kwargs):
        N = kwargs['N']
        batch_size = kwargs['batch_size']
        bias = (N-1) / (batch_size-1)
        sample_dim = 0
        batch_dim = 1
        z = [n for n in q.sampled if n in p]
        z = torch.cat(z, -1)
        mis = []
        for i in range(z.size()[-1]):
            log_qz = q.log_joint(sample_dim, batch_dim, z[..., i])
            log_joint_avg_qz, _, _ = q.log_batch_marginal(sample_dim, batch_dim, z[..., i], bias=bias)
            mis.append((log_qz - log_joint_avg_qz).mean())

        return mis
