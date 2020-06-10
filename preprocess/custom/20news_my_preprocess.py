import gzip
from preprocess.vectorizer import Vectorizer
import numpy as np
import json
from scipy import sparse
import os
from preprocess.text_preprocessing import clean_text

def load_from_json(path):
    data = json.load(open(path,'r'))
    x = [d['text'] for d in data]
    y = [d['label'] for d in data]
    return x, y

if __name__ == '__main__':
    os.chdir('../..')
    TRAIN_PATH = 'resources/datasets/20_newsgroups_old/json/train.json'
    TEST_PATH = 'resources/datasets/20_newsgroups_old/json/test.json'
    SAVE_PATH = 'resources/datasets/20news/new/'
    CUTOFF = 3 # cutoff examples which have less than CUTOFF words in BoW after preprocessing
    X_train, Y_train = load_from_json(TRAIN_PATH)
    X_test, Y_test = load_from_json(TEST_PATH)
    print('Found %d documents in train and %d documents in test'%(len(X_train), len(X_test)))
    X_raw = [clean_text(x) for x in X_train + X_test]
    json.dump(X_raw, open(SAVE_PATH + 'X_raw.json', 'w'))

    vectorizer = Vectorizer(vocab_size=2000)
    vectorizer.build_vocab(X_train, save_path=SAVE_PATH+'vocab_full.json')
    # vectorizer.load_vocab(SAVE_PATH+'vocab_full.json')
    X_train = vectorizer.text_to_bow(X_train)
    X_test = vectorizer.text_to_bow(X_test)

    train_inds = np.where(X_train.sum(-1)>=CUTOFF)[0]
    X_train, Y_train = X_train[train_inds], np.array(Y_train)[train_inds]
    test_inds = np.where(X_test.sum(-1)>=CUTOFF)[0]
    X_test, Y_test = X_test[test_inds], np.array(Y_test)[test_inds]

    sparse.save_npz(SAVE_PATH+'train.npz', X_train)
    sparse.save_npz(SAVE_PATH+'test.npz', X_test)

    # save y, labels
    labels = {y:i for (i,y) in enumerate(sorted(set(Y_train)))}
    json.dump(list(labels.keys()), open(SAVE_PATH+'labels.json','w'))
    Y_train, Y_test = np.array([labels[y] for y in Y_train]), np.array([labels[y] for y in Y_test])
    np.save(SAVE_PATH+'train.labels.npy', Y_train)
    np.save(SAVE_PATH+'test.labels.npy', Y_test)





