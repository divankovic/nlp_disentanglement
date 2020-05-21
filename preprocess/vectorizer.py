import json
from math import ceil

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from preprocess.text_preprocessing import clean_text


class Vectorizer:
    """
    Class for fitting a vocabulary and vectorizing texts.
    Used for extracting BoW or word2vec features.
    """

    def __init__(self, min_seq_length=10, max_sequence_length=30, min_occ=1, unknown='unk'):
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_seq_length
        self.min_occ = min_occ
        self.unknown = 'unk'

    def load_vocab(self, vocab_path):
        print('Loading vocabulary from %s' % vocab_path)
        with open(vocab_path, 'r') as vocab_file:
            vocab = json.load(vocab_file)
        self.word2idx = {word: int(idx) for word, idx in vocab['word2idx'].items()}
        self.idx2word = {int(idx): word for idx, word in vocab['idx2word'].items()}
        self.vocab_size = len(self.word2idx)

    def build_vocab(self, data, vocab_path):
        count_vectorizer = CountVectorizer(tokenizer=clean_text, min_df=self.min_occ, token_pattern=None)
        count_vectorizer.fit(data)
        self.word2idx = count_vectorizer.vocabulary_

        for word in self.word2idx:
            self.word2idx[word] += 1
        self.word2idx[self.unknown] = 0
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        with open(vocab_path, 'w') as vocab_file:
            vocab = {'word2idx': self.word2idx, 'idx2word': self.idx2word}
            json.dump(vocab, vocab_file)
            print('Saving vocabulary to %s' % vocab_path)

    def extract_embeddings(self, model):
        """
        Extracts word embeddings from the data using the model.
        Note : if the word is not in the model, then it is randomly initalized.
        Parameters
        ----------
        model - word to embedding model (gensim model)
        """
        print('Loading embeddings...')
        self.embedding_size = model.vector_size
        self.embeddings = np.zeros([self.vocab_size, self.embedding_size])
        found = 0
        for i, word in sorted(self.idx2word.items()):
            if word in model:
                self.embeddings[i] = model[word]
                found += 1
            else:
                # initialize randomly
                self.embeddings[i] = np.random.randn(self.embedding_size)

        print('Found %d words in the embedding model out of %d in the vocabulary.' % (found, self.vocab_size))

    def text_to_sequences(self, data, pad=True, maxlen_ratio=None):
        texts_tokenized = list(
            map(lambda s: [self.unknown if word not in self.word2idx else word for word in clean_text(s)],
                data))
        sequences = list(map(lambda s: [self.word2idx[word] for word in s], texts_tokenized))

        if maxlen_ratio:
            # ignore self.max_sequence_length and use the ratio
            lengths = [len(text) for text in texts_tokenized]
            self.max_sequence_length = int(ceil(np.percentile(lengths, maxlen_ratio * 100)))
            print('Cutting off sequences at %dth percentile, which is %d words' % (
                maxlen_ratio*100, self.max_sequence_length))
        sequences = [seq[:self.max_sequence_length] for seq in sequences if len(seq) > self.min_sequence_length]

        if pad:
            for sequence in sequences:
                sequence.extend([self.word2idx[self.unknown]] * (self.max_sequence_length - len(sequence)))

        return sequences

    def text_to_embeddings(self, data, pad=True, maxlen_ratio=None, aggregation='mean'):
        """
        Sequence features are obtained by aggregating the embeddings of the words.
        Aggregation can either be 'stack' (return type N x seq_len x embedding_size) or 'mean' (return N x embedding_size)

        Returns
        -------
        embeddings for the data
        """
        print('Converting texts to embeddings')
        sequences = self.text_to_sequences(data, pad=pad, maxlen_ratio=maxlen_ratio)

        vectors = []
        for sequence in sequences:
            sequence_embedding = [self.embeddings[i] for i in sequence]
            vectors.append(sequence_embedding)

        vectors = np.array(vectors)
        if aggregation == 'mean':
            # vectors shape here is N x seq_len x embedding_size
            vectors = np.mean(vectors, axis=1)

        return vectors
