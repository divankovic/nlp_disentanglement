from models.vae import VAE
import probtorch
from torch import nn


# ProbTorch VAE implementations

# vae implemetation modified to be compatible with probtorch
class PTVAE(nn.Module):
    def __init__(self, encoder, decoder, num_samples=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_samples = num_samples

    def forward(self, x):
        q = self.encoder(x)
        p = self.decoder(x, q)
        return q, p

    def loss_function(self, q, p, **kwargs):
        return - probtorch.objectives.montecarlo.elbo(q, p, sample_dim=0, batch_dim=1, beta=1.0)


class HFVAE(PTVAE):

    def __init__(self, encoder, decoder, beta=None, num_samples=1):
        super().__init__(encoder, decoder, num_samples)
        if beta is None:
            beta = (1.0, 1.0, 1.0, 1.0, 1.0)
        self.beta = beta

    def loss_function(self, q, p, **kwargs):
        N = kwargs['N']
        batch_size = kwargs['batch_size']
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.0
        NUM_SAMPLES = 1
        bias = (N - 1) / (batch_size - 1)
        if NUM_SAMPLES is None:
            # wont work is NUM_SAMPLES is None
            return -probtorch.objectives.marginal.elbo(q, p, sample_dim=None, batch_dim=0, alpha=alpha, beta=self.beta,
                                                       bias=bias)
        else:
            return -probtorch.objectives.marginal.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha, beta=self.beta,
                                                       bias=bias)

    # TODO : add mutual information function I(x, z) -- maybe also reconstruct I(x, zd) from it