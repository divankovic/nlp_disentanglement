from models.vae import VAE
import probtorch


class HFVAE(VAE):

    def __init__(self, encoder, decoder, beta=None):
        super(HFVAE, self).__init__(encoder, decoder)
        if beta is None:
            beta = (1.0, 1.0, 1.0, 1.0, 1.0)
        self.beta = beta

    def forward(self, x, **kwargs):
        q = self.encoder(x)
        p = self.decoder(x, q)
        return q, p

    def loss_function(self, q, p, N, batch_size, alpha=0.0, NUM_SAMPLES=1):
        bias = (N - 1) / (batch_size - 1)
        # hint : for classic NVDM the loss function used should be
        # probtorch.objectives.montecarlo.elbo(q,p,beta=1.0)
        if NUM_SAMPLES is None:
            # wont work is NUM_SAMPLES is None
            return -probtorch.objectives.marginal.elbo(q, p, sample_dim=None, batch_dim=0, alpha=alpha, beta=self.beta,
                                                       bias=bias)
        else:
            return -probtorch.objectives.marginal.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha, beta=self.beta,
                                                       bias=bias)
