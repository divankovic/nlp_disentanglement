from abc import abstractmethod, ABC

import torch
from torch import nn
import torch.nn.init as init
import gensim

from utils.vae_utils import reparametrize, reconstruction_loss, kl_divergence


class BaseVAE(nn.Module):
    def __init__(self, encoder, decoder, recon_distribution=None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.recon_distribution = recon_distribution

    def encode(self, x, **kwargs):
        return self.encoder(x)

    def decode(self, z, **kwargs):
        return self.decoder(z)

    def sample_latent(self, x):
        mu, logvar = self.encode(x)
        z = reparametrize(mu, logvar)
        return z.cpu().detach().numpy(), mu.cpu().detach().numpy()

    def forward(self, x, **kwargs):
        mu, logvar = self.encode(x)
        z = reparametrize(mu, logvar)
        return self.decode(z).view(x.size()), mu, logvar

    def loss_function(self, x_recon, x, mu, logvar):
        components = self.loss_components(x_recon, x, mu, logvar)
        return components['reconstruction_loss'] + components['kl_divergence']

    def loss_components(self, x_recon, x, mu, logvar):
        return {'reconstruction_loss': reconstruction_loss(x_recon, x, self.recon_distribution),
                'kl_divergence': kl_divergence(mu, logvar)}


class SequenceVAE(BaseVAE):
    # Text generation VAE
    def __init__(self, encoder, decoder, vocab_size, dropout, recon_distribution='categorical', embeddings=None):
        # embeddings - path to pretrained embeddings
        # if none, custom embeddings will be trained on the dataset
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.embedding_size = encoder.input_dim
        self.vocab_size = vocab_size
        self.recon_distribution = recon_distribution
        self.drop = nn.Dropout(dropout)

        if embeddings:
            model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/biowordvec_pubmed_d200.vec.bin',
                                                                    binary=True)
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(model.vectors))
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

    def forward(self, x, **kwargs):
        """
        Parameters
        ----------
        x - tensor of shape [batch_size, seq_len]
        kwargs - must have an embedding model_0 under key 'embedding'
        """
        x = self.drop(self.embedding(x))
        mu, logvar = self.encode(x)
        z = reparametrize(mu, logvar)
        logits, _ = self.decoder(x, z, self.drop)
        return logits, mu, logvar
