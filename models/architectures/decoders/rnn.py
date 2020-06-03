import torch.nn as nn
from torch import softmax


# implemented for binary cross entropy loss with word indices in the vocabulary
# alternative - reconstruct the word embeddings and use MSE as the error
#             - reconstruct the word embeddings, find the nearest embedding and use that word, again binary CE loss
class RNNDecoder(nn.Module):
    def __init__(self, latent_dim, input_dim, vocab_size, rnn_type, hidden_size, n_layers, dropout=0.2,
                 bidirectional=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim  # usually word embedding size
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        if rnn_type not in ['RNN', 'GRU', 'LSTM']:
            raise ValueError('rnn type %s not supported! Must be one of [RNN, GRU, LSTM]' % rnn_type)

        self.hidden_factor = (2 if bidirectional else 1)
        # self.latent2hidden = nn.Linear(latent_dim, hidden_size * self.hidden_factor)
        self.rnn = getattr(nn, rnn_type)(input_size=input_dim, hidden_size=hidden_size, num_layers=n_layers,
                                         batch_first=True, dropout=dropout if n_layers>1 else 0, bidirectional=bidirectional)
        self.latent2emb = nn.Linear(latent_dim, input_dim)
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, z, dropout, hidden=None):
        """
        Parameters
        ----------
        x - [batch_size, seq_len, embedding_size] tensor
        z - latent context : [batch_size, latent_dim] tensor

        Returns
        -------
        Unnormalized logits of sentence words distribution probabilities of
        shape [batch_size, seq_len, vocab_size]
        """
        x = x + self.latent2emb(z)
        if self.rnn_type == 'LSTM':
            output, (hidden, _) = self.rnn(x, hidden)
        else:
            output, hidden = self.rnn(x, hidden)

        output = dropout(output)
        logits = self.hidden2vocab(output.view(-1, output.size(-1))).view(output.size(0), output.size(1), -1)
        return logits, hidden