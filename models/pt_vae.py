import probtorch
from torch import nn
import numpy as np


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

    def sample_latent(self, x):
        q = self.encoder(x)
        z = q['z'].value.cpu().detach().numpy()
        z_mu = q['z'].dist.mean.cpu().detach().numpy()
        return z, z_mu

    def loss_function(self, q, p, reduce=True, **kwargs):
        return - probtorch.objectives.montecarlo.elbo(q, p, sample_dim=0, batch_dim=1, beta=1.0, reduce=reduce)

    def loss_components(self, q, p, **kwargs):
        sample_dim = 0
        batch_dim = 1
        log_weights = q.log_joint(sample_dim, batch_dim, q.conditioned())
        losses = {'log_like': - probtorch.objectives.montecarlo.log_like(q, p, sample_dim, batch_dim, log_weights,
                                                                         size_average=True, reduce=True),
                  'kl_divergence': probtorch.objectives.montecarlo.kl(q, p, sample_dim, batch_dim, log_weights,
                                                                      size_average=True, reduce=True)}
        return losses


class HFVAE(PTVAE):

    def __init__(self, encoder, decoder, beta=None):
        super().__init__(encoder, decoder)
        if beta is None:
            # beta template for hfvae (gammma, 1, alpha, beta, 0)
            beta = (1.0, 1.0, 1.0, 1.0, 0)
        self.beta = beta

    def sample_latent(self, x):
        q = self.encoder(x)
        z_groups = []
        z_mu_groups = []
        for i in range(self.encoder.num_groups):
            z = q['z_' + str(i)].value.cpu().detach().numpy()
            mu = q['z_' + str(i)].dist.mean.cpu().detach().numpy()
            z_groups.append(z)
            z_mu_groups.append(mu)
        return np.concatenate(z_groups, axis=-1), np.concatenate(z_mu_groups, axis=-1)

    def loss_function(self, q, p, reduce=True, **kwargs):
        N = kwargs['N']
        batch_size = kwargs['batch_size']
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.0
        bias = (N - 1) / (batch_size - 1)
        return -probtorch.objectives.marginal.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha, beta=self.beta,
                                                   bias=bias, reduce=reduce)

    def loss_components(self, q, p, **kwargs):
        N = kwargs['N']
        batch_size = kwargs['batch_size']
        bias = (N - 1) / (batch_size - 1)
        sample_dim = 0
        batch_dim = 1
        losses = {}

        log_weights = q.log_joint(sample_dim, batch_dim, q.conditioned())
        losses['log_like(1+3)'] = -probtorch.objectives.montecarlo.log_like(q, p, sample_dim, batch_dim, log_weights,
                                                                            size_average=True, reduce=True)

        z = [n for n in q.sampled() if n in p]
        log_joint_avg_pz, log_avg_pz, log_avg_pzd_prod = p.log_batch_marginal(sample_dim, batch_dim, z, bias=1.0)
        log_joint_avg_qz, log_avg_qz, log_avg_qzd_prod = q.log_batch_marginal(sample_dim, batch_dim, z, bias=bias)
        log_pz = p.log_joint(sample_dim, batch_dim, z)
        log_qz = q.log_joint(sample_dim, batch_dim, z)
        losses['mutual_information(2)'] = self.beta[2] * (log_qz - log_joint_avg_qz).mean()
        losses['tc_1(A)'] = self.beta[3] * ((log_joint_avg_qz - log_avg_qz) - (log_joint_avg_pz - log_pz)).mean()
        losses['tc_2(i)'] = self.beta[0] * ((log_avg_qz - log_avg_qzd_prod) - (log_pz - log_avg_pzd_prod)).mean()
        losses['kl_to_prior(ii)'] = self.beta[1] * (log_avg_qzd_prod - log_avg_pzd_prod).mean()

        return losses

    # TODO - test this out
    def mutual_info_by_components(self, q, p, **kwargs):
        N = kwargs['N']
        batch_size = kwargs['batch_size']
        bias = (N - 1) / (batch_size - 1)
        sample_dim = 0
        batch_dim = 1
        z = [n for n in q.sampled() if n in p]
        z = torch.cat(z, -1)
        mis = []
        for i in range(z.size()[-1]):
            log_qz = q.log_joint(sample_dim, batch_dim, z[..., i])
            log_joint_avg_qz, _, _ = q.log_batch_marginal(sample_dim, batch_dim, z[..., i], bias=bias)
            mis.append((log_qz - log_joint_avg_qz).mean())
