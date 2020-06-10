import torch.nn as nn
import torch.backends.cudnn as cudnn
import gensim

import random
import numpy as np
import torch


def set_seed(seed):  # set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Unsqueeze3D(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


# TODO - alternatively - train own embeddings, or a nn embedding layer which is trained together with the network
def load_embeddings(path, binary=True):
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)
    return model
