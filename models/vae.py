from abc import abstractmethod, ABC

import torch
from torch import nn
import torch.nn.init as init

from utils.vae_utils import reparametrize, reconstruction_loss, kl_divergence


class VAE(nn.Module):
    def __init__(self, encoder, decoder, recon_distribution='bernoulli'):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.recon_distribution = recon_distribution

    def encode(self, x, **kwargs):
        return self.encoder(x)

    def decode(self, z, **kwargs):
        return self.decoder(z)

    def init_layers(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    init.xavier_normal_(m.weight.data)
                elif isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight.data)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                else:
                    init.xavier_normal_(m.weight.data)

    def forward(self, x, **kwargs):
        mu, logvar = self.encode(x)
        z = reparametrize(mu, logvar)
        return self.decode(z).view(x.size()), mu, logvar

    def loss_function(self, x_recon, x, mu, logvar):
        return reconstruction_loss(x_recon, x, self.recon_distribution) + kl_divergence(mu, logvar)

    def sample(self):
        # implement sampling
        pass


class SequenceVAE(VAE):
    # Text generation VAE
    # TODO - come back to this after implementing and experimenting with simpler stuff on text (fc, cnn, ...)
    # guideline - text-autoencoders repo
    def __init__(self, encoder, decoder, recon_distribution='categorical'):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.recon_distribution = recon_distribution

    def forward(self, x, **kwargs):
        """
        Parameters
        ----------
        x - tensor of shape [batch_size, seq_len]
        kwargs - must have an embedding model_0 under key 'embedding'
        """
        embedding = kwargs['embedding']
        x_embedded = embedding(x)
        mu, logvar = self.encode(x_embedded)
        z = reparametrize(mu, logvar)
        output = self.decoder(x_embedded, z)
        return output, mu, logvar

