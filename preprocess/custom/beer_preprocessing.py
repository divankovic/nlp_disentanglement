import gzip
from preprocess.vectorizer import Vectorizer
import numpy as np
import json
from scipy import sparse
import os
from preprocess.text_preprocessing import clean_text

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
    CUTOFF = 3
    X_train, Y_train = beer_reader(TRAIN_PATH)
    X_test, Y_test = beer_reader(TEST_PATH)
    X_train = [' '.join(x) for x in X_train]
    X_test = [' '.join(x) for x in X_test]
    X_raw = [clean_text(x) for x in X_train + X_test]
    json.dump(X_raw, open(SAVE_PATH + 'X_raw.json', 'w'))

    vectorizer = Vectorizer(vocab_size=20000)
    vectorizer.build_vocab(X_train, save_path=SAVE_PATH+'vocab_full.json')
    #vectorizer.load_vocab(SAVE_PATH+'vocab_full.json')
    X_train = vectorizer.text_to_bow(X_train)
    X_test = vectorizer.text_to_bow(X_test)

    train_inds = np.where(X_train.sum(-1) >= CUTOFF)[0]
    X_train, Y_train = X_train[train_inds], Y_train[train_inds]
    test_inds = np.where(X_test.sum(-1) >= CUTOFF)[0]
    X_test, Y_test = X_test[test_inds], Y_test[test_inds]

    sparse.save_npz(SAVE_PATH+'train.npz', X_train)
    sparse.save_npz(SAVE_PATH+'test.npz', X_test)
    np.save(SAVE_PATH + 'train.labels.npy', Y_train)
    np.save(SAVE_PATH + 'test.labels.npy', Y_test)





