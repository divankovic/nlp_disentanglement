import torch.nn as nn
import torch


class RNNEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, rnn_type, hidden_size, n_layers, dropout=0.2, bidirectional=False,
                 embedding_layer=False):
        super().__init__()
        self.input_dim = input_dim  # usually word embeddings size
        self.latent_dim = latent_dim
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        if rnn_type not in ['RNN', 'GRU', 'LSTM']:
            raise ValueError('rnn type %s not supported! Must be one of [RNN, GRU, LSTM]' % rnn_type)
        # note - nonlinearity can be changed for vanilla rnn
        self.rnn = getattr(nn, rnn_type)(input_size=input_dim, hidden_size=hidden_size, num_layers=n_layers,
                                         batch_first=True, dropout=dropout if n_layers > 1 else 0,
                                         bidirectional=bidirectional)

        self.hidden_factor = (2 if bidirectional else 1)
        self.mu = nn.Linear(hidden_size * self.hidden_factor, latent_dim)
        self.logvar = nn.Linear(hidden_size * self.hidden_factor, latent_dim)

    def forward(self, x):
        """
        Parameters
        ----------
        x - [batch_size, seq_len, embedding_size] tensor
        """
        batch_size, seq_len, embedding_size = x.size()
        if self.rnn_type == 'LSTM':
            output, (hidden, _) = self.rnn(x)
        else:
            output, hidden = self.rnn(input)

        hidden = torch.cat([hidden[-2], hidden[-1]], 1)
        return self.mu(hidden), self.logvar(hidden)


class TextModel(nn.Module):
    def __init__(self, vocab, args, initrange=0.1):
        pass
