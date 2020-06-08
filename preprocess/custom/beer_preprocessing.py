import gzip
from preprocess.vectorizer import Vectorizer
import numpy as np
import json
from scipy import sparse
import os

def beer_reader(path, max_len=0):
    """
    Reads in beer multi-aspect sentiment raw.
    Parameters
    ----------
    path - path to file
    Returns
    """
    scores = []
    tokens = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            scores.append(list(map(float, parts[:5])))
            toks = parts[5:]
            if max_len > 0:
                toks = toks[:max_len]
            tokens.append(toks)

    return tokens, scores


if __name__ == '__main__':
    os.chdir('../..')
    TRAIN_PATH = 'resources/datasets/beer_reviews/raw/reviews.260k.train.txt.gz'
    TEST_PATH = 'resources/datasets/beer_reviews/raw/reviews.260k.heldout.txt.gz'
    SAVE_PATH = 'resources/datasets/beer_reviews/'
    X_train, Y_train = beer_reader(TRAIN_PATH)
    X_test, Y_test = beer_reader(TEST_PATH)

    np.save(SAVE_PATH + 'train.labels.npy', Y_train)
    np.save(SAVE_PATH + 'test.labels.npy', Y_test)
    json.dump(X_train + X_test, open(SAVE_PATH + 'X_raw.json', 'w'))

    X_train = [' '.join(x) for x in X_train]
    X_test = [' '.join(x) for x in X_test]
    vectorizer = Vectorizer(vocab_size=20000)
    vectorizer.build_vocab(X_train, save_path=SAVE_PATH+'vocab.json')
    #vectorizer.load_vocab(SAVE_PATH+'vocab.json')
    X_train = vectorizer.text_to_bow(X_train)
    X_test = vectorizer.text_to_bow(X_test)

    sparse.save_npz(SAVE_PATH+'train.npz', sparse.csr_matrix(X_train))
    sparse.save_npz(SAVE_PATH+'test.npz', sparse.csr_matrix(X_test))
    # np.save(save_path, X_train)
    # np.save(save_path, X_test)




