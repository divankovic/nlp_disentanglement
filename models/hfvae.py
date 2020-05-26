from models.vae import VAE
import probtorch


class HFVAE(VAE):

    def __init__(self, encoder, decoder):
        super(HFVAE, self).__init__(encoder, decoder)

    def forward(self, x, **kwargs):
        q = self.encoder(x)
        p = self.decoder(x, q)
        return q, p

    def loss_function(self, q, p, N, B, alpha=0.1, NUM_SAMPLES=1):
        # beta = () - leave default for now
        bias = (N - 1) / (B - 1)
        if NUM_SAMPLES is None:
            return -probtorch.objectives.marginal.elbo(q, p, sample_dim=None, batch_dim=0, alpha=alpha, bias=bias)
        else:
            return -probtorch.objectives.marginal.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha, bias=bias)
