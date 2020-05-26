import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


# standard case - normal distrib, extend if neccessary
def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def reconstruction_loss(x_recon, x, distribution):
    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    elif distribution == 'categorical':
        recon_loss = F.cross_entropy(x_recon, x)
    elif distribution == 'gauss':
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    else:
        raise ValueError('only bernoulli and gaussian supported for reconstruction loss, received : %s' % distribution)

    return recon_loss


# TODO - extend to other continuous and discrete distributions
def kl_divergence(mu, logvar):
    # TODO manually calculated for diagonal normal prior
    # consider extending with probabilistic programming libraries
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
