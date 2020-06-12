import itertools
import pickle
from collections import defaultdict
from math import log
import sys

import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import Dense2Corpus
from gensim.models.coherencemodel import CoherenceModel
from utils.file_handling import MultiOutput


def get_topics(beta, idx2word, n_top=10):
    return [[idx2word[j] for j in beta[i].argsort()[:-n_top - 1:-1]] for i in range(len(beta))]


def print_top_words(beta, idx2word, n_top=10, save_path=None):
    """
    Print top n_top_words for each topic.

    Parameters
    ----------
    beta - weights of the decoder (word distribution over topics) in the form of (latent_dim, vocab_dim)
    idx2word - index to word mapping from the vocabulary
    n_top_words - top words to print for each latent variable (topic)

    Returns
    -------
    """
    if save_path:
        log = MultiOutput(sys.stdout, open(save_path, 'w'))
    else:
        log = MultiOutput(sys.stdout)

    log.print('--------------- Topics ------------------')
    topics = get_topics(beta, idx2word, n_top=n_top)
    for topic in topics:
        log.print(' '.join(topic))
    log.print('-----------------------------------------')


def umass_coherence_score(model, X, score_num=20):
    """
    Computes the coherence score of the learned topics using gensim's implementation.
    Parameters
    ----------
    model - trainedmodel
    X - bow input
    score_num - top words in topic hiperparameter
    Returns
    -------
    calculated coherence score using npmi estimate
    """

    model.eval()
    decoder_weight = model.decoder.main[0].weight.detach().cpu()
    corpus = Dense2Corpus(X, documents_columns=False)
    id2word = {index: str(index) for index in range(X.shape[1])}
    topics = [
        [str(item.item()) for item in topic]
        for topic in decoder_weight.topk(min(score_num, X.shape[1]), dim=0)[1].t()
    ]

    coherence_model = CoherenceModel(topics=topics, corpus=corpus, dictionary=Dictionary.from_corpus(corpus, id2word),
                                     coherence='u_mass')
    # return coherence_model.get_coherence()
    return coherence_model.get_coherence_per_topic()


def npmi_coherence_score_gensim(model, X_raw, X, idx2word, score_num=20):
    """
    Computes the coherence score of the learned topics using gensim's implementation.
    Parameters
    ----------
    model - trainedmodel
    X_raw - tokenized texts
    X - bow input
    idx2word - mapping from index to word from the vocabulary
    score_num - top words in topic hiperparameter

    Returns
    -------
    calculated coherence score using npmi estimate
    """
    # bow to indices
    # data_ind = np.array([np.where(x == a_0)[0] for x in X])
    # X_raw = np.array([[idx2word[i] for i in x] for x in data_ind])

    model.eval()
    decoder_weight = model.decoder.main[0].weight.detach().cpu()
    corpus = Dense2Corpus(X, documents_columns=False)
    topics = [
        [idx2word[item.item()] for item in topic]
        for topic in decoder_weight.topk(min(score_num, X.shape[1]), dim=0)[1].t()
    ]

    coherence_model = CoherenceModel(topics=topics, texts=X_raw, corpus=corpus,
                                     dictionary=Dictionary.from_corpus(corpus, idx2word),
                                     coherence='c_npmi', window_size=0)
    # return coherence_model.get_coherence()
    return coherence_model.get_coherence_per_topic()


def calculate_word_frequencies(corpus, vocab, save_path):
    if save_path is None:
        raise ValueError('save_path must be provided')
    word_frequencies = defaultdict(int)
    joint_word_frequencies = defaultdict(int)
    N = len(corpus)
    for i in range(len(corpus)):
        print('Processing document %d out of %d' % (i, N))
        doc = corpus[i]
        doc_words = set(doc)
        doc_words = [word for word in doc_words if word in vocab]
        for word in doc_words:
            word_frequencies[word] += 1
        for wi, wj, in itertools.combinations(doc_words, 2):
            joint_word_frequencies[(wi, wj)] += 1
            joint_word_frequencies[(wj, wi)] += 1

    word_frequencies = {k: v / N for k, v in word_frequencies.items()}
    joint_word_frequencies = {k: v / N for k, v in joint_word_frequencies.items()}
    pickle.dump(word_frequencies, open(save_path + '/word_frequencies.pkl', 'wb'))
    pickle.dump(joint_word_frequencies, open(save_path + '/jointword_frequencies.pkl', 'wb'))
    return word_frequencies, joint_word_frequencies


def npmi_coherence_score(topics, word_frequencies, joint_word_frequencies):
    """
    Compute npmi score. Implemented guided by https://arxiv.org/abs/1812.05035, appendix A.
    Average word similarity between all pairs of words in a topic. It is bounded by [-a_0, a_0]
    Parameters
    ----------
    topics - (n_topics, top_words) array
    corpus - 2d array or list of lists - tokenized sequences of words in the corpus
    word_frequencies={w_i : p_wi}
    joint_word_frequencies={(w_i, w_j):p_wi_wj}

    Returns
    -------
    NPMI scores per topic
    """
    topics = np.array(topics)
    K = topics.shape[0]  # number of topics
    T = topics.shape[1]  # top words in a topic

    topic_coherences = []
    for k in range(K):
        topic_coherence = 0
        for w_i, w_j in itertools.combinations(topics[k], 2):
            # npmi for each word pair in a topic
            p_wiwj = joint_word_frequencies[(w_i, w_j)] if (w_i, w_j) in joint_word_frequencies else 0
            if (p_wiwj - 0) < 1e-12:
                # the joint frequency is zero, npmi is -a_0 : https://stats.stackexchange.com/questions/140935/how-does-the-logpx-y-normalize-the-point-wise-mutual-information
                topic_coherence += -1
                continue
            p_wi = word_frequencies[w_i]
            p_wj = word_frequencies[w_j]
            topic_coherence += log(p_wiwj / (p_wi * p_wj)) / (-log(p_wiwj))

        topic_coherence = topic_coherence / (T * (T - 1) / 2)
        topic_coherences.append(topic_coherence)

    return topic_coherences


def get_most_correlated_topics(cov_matrix,top_correlations=4):
    latent_dim = cov_matrix.shape[0]
    inds = np.dstack(np.unravel_index(np.argsort(cov_matrix.ravel()), (latent_dim, latent_dim)))[0]
    inds = inds[-latent_dim- top_correlations:-latent_dim, :]
    covs = [cov_matrix[tuple(ind)] for ind in inds]
    cor_topics = sorted(set(inds.flatten()))

    return inds, covs, cor_topics
