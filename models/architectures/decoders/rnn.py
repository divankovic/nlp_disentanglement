import torch.nn as nn
from torch import softmax


# implemented for binary cross entropy loss with word indices in the vocabulary
# alternative - reconstruct the word embeddings and use MSE as the error
#             - reconstruct the word embeddings, find the nearest embedding and use that word, againd binary CE loss
class RNNDencoder(nn.Module):
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

        self.hidden_factor = (2 if bidirectional else 1) * n_layers
        self.latent2hidden = nn.Linear(latent_dim, hidden_size * self.hidden_factor)
        self.rnn = getattr(nn, rnn_type)(input_size=input_dim, hidden_size=hidden_size, num_layers=n_layers,
                                         batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.outputs2vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, z):
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
        hidden = self.latent2hidden(z)
        batch_size = z.size()[0]
        seq_len = x.size()[1]

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        outputs, _ = self.rnn(x, hidden)
        outputs = outputs.view(-1, self.hidden_size)
        # project outputs to vocab
        result = softmax(self.outputs2vocab(outputs))

        return result.view(batch_size, seq_len, self.vocab_size)
