import json
from math import ceil
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from preprocess.text_preprocessing import clean_text


class Vectorizer:
    """
    Class for fitting a vocabulary and vectorizing texts.
    Used for extracting BoW or word2vec features.
    """

    def __init__(self, min_seq_length=10, max_sequence_length=30, min_occ=1, vocab_size=None, unknown='unk'):
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_seq_length
        self.min_occ = min_occ
        self.unknown = 'unk'
        self.vocab_size = vocab_size

    def load_vocab(self, vocab_path):
        print('Loading vocabulary from %s' % vocab_path)
        with open(vocab_path, 'r') as vocab_file:
            vocab = json.load(vocab_file)
        self.word2idx = {word: int(idx) for word, idx in vocab['word2idx'].items()}
        self.idx2word = {int(idx): word for idx, word in vocab['idx2word'].items()}
        self.vocab_size = len(self.word2idx)
        self.count_vectorizer = CountVectorizer(tokenizer=clean_text, vocabulary=self.word2idx)

    def build_vocab(self, data, save_path):
        count_vectorizer = CountVectorizer(tokenizer=clean_text, min_df=self.min_occ, token_pattern=None
                                           , max_features=self.vocab_size)
        count_vectorizer.fit(data)
        self.count_vectorizer = count_vectorizer  # for extracting BoW features
        self.word2idx = count_vectorizer.vocabulary_

        # for word in self.word2idx:
        #    self.word2idx[word] += 1
        # self.word2idx[self.unknown] = 0
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        if save_path:
            with open(save_path, 'w') as vocab_file:
                vocab = {'word2idx': self.word2idx, 'idx2word': self.idx2word}
                json.dump(vocab, vocab_file)
                print('Saving vocabulary to %s' % save_path)

    def text_to_bow(self, data, binary=False):
        data = self.count_vectorizer.transform(data)
        if binary:
            data[data != 0] = 1
        return data

    def extract_embeddings(self, model):
        """
        Extracts word embeddings from the raw using the model_0.
        Note : if the word is not in the model_0, then it is randomly initalized.
        Parameters
        ----------
        model_0 - word to embedding model_0 (gensim model_0)
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

        print('Found %d words in the embedding model_0 out of %d in the vocabulary.' % (found, self.vocab_size))

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
                maxlen_ratio * 100, self.max_sequence_length))
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
        embeddings for the raw
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


class SimpleSequenceVocab:
    """
    Used for experiments with text-generation setup.
    No text preprocessing
    """

    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []

        with open(path) as f:
            for line in f:
                w = line.split()[0]
                self.word2idx[w] = len(self.word2idx)
                self.idx2word.append(w)
        self.size = len(self.word2idx)

        self.pad = self.word2idx['<pad>']
        self.go = self.word2idx['<go>']
        self.eos = self.word2idx['<eos>']
        self.unk = self.word2idx['<unk>']
        self.blank = self.word2idx['<blank>']

    @staticmethod
    def build(sents, path, size):
        v = ['<pad>', '<go>', '<eos>', '<unk>', '<blank>']
        words = [w for s in sents for w in s]
        cnt = Counter(words)
        n_unk = len(words)
        for w, c in cnt.most_common(size):
            v.append(w)
            n_unk -= c
        cnt['<unk>'] = n_unk

        with open(path, 'w') as f:
            for w in v:
                f.write('{}\t{}\n'.format(w, cnt[w]))

    @staticmethod
    def strip_eos(sents):
        return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
                for sent in sents]
