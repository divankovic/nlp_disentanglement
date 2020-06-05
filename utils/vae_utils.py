import torch
import torch.nn.functional as F


def reparametrize(mu, logvar):
    # standard case - normal distribution
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def reconstruction_loss(x_recon, x, distribution=None):
    if distribution is None:
        recon_loss = -torch.sum((torch.log(x_recon)*x))
    elif distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    elif distribution == 'categorical':
        recon_loss = F.cross_entropy(x_recon, x)
    elif distribution == 'gauss':
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    else:
        raise ValueError('only bernoulli and gaussian supported for reconstruction loss, received : %s' % distribution)

    return recon_loss


def kl_divergence(mu, logvar):
    # manually calculated for diagonal normal prior
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
