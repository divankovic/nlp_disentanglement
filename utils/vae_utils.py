import torch
import torch.nn.functional as F


def reparametrize(mu, logvar):
    # standard case - normal distribution
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def reconstruction_loss(x_recon, x, distribution=None):
    if distribution is None:
        recon_loss = -(torch.log(x_recon)*x).sum(1).mean()
    elif distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')
    elif distribution == 'categorical':
        recon_loss = F.cross_entropy(x_recon, x)
    elif distribution == 'gauss':
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    else:
        raise ValueError('only bernoulli and gaussian supported for reconstruction loss, received : %s' % distribution)

    return recon_loss


def kl_divergence(mu, logvar):
    # manually calculated for diagonal normal prior
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
