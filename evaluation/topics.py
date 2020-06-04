from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import Dense2Corpus
import numpy as np
import itertools
from math import log


def print_top_words(beta, idx2word, n_top_words=10):
    """
    Print top n_top_words for each topic.

    Parameters
    ----------
    beta - weights of the decoder in the form of (latent_dim, vocab_dim)
    idx2word - index to word mapping from the vocabulary
    n_top_words - top words to print for each latent variable (topic)

    Returns
    -------
    """
    print('--------------- Topics ------------------')
    for i in range(len(beta)):
        print(' '.join([idx2word[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
    print('-----------------------------------------')


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
    # data_ind = np.array([np.where(x == 1)[0] for x in X])
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
                                     coherence='c_v')
    # return coherence_model.get_coherence()
    return coherence_model.get_coherence_per_topic()


def calculate_word_frequencies(topics, corpus):
    word_frequencies = {}
    joint_word_frequencies = {}
    words = set(np.array(topics).flatten())

    N = len(corpus)
    for w_i in words:
        N_i = sum(map(lambda text: 1 if w_i in text else 0, corpus))
        p_wi = N_i / N
        word_frequencies[w_i] = p_wi

    for w_i, w_j in itertools.combinations(words):
        N_ij = sum(map(lambda text: 1 if (w_i in text and w_j in text) else 0))
        p_wiwj = N_ij / N
        joint_word_frequencies[(w_i, w_j)] = p_wiwj
        joint_word_frequencies[(w_j, w_i)] = p_wiwj

    return word_frequencies, joint_word_frequencies


def npmi_coherence_score(topics, corpus):
    """
    Compute npmi score. Implemented guided by https://arxiv.org/abs/1812.05035, appendix A.
    Parameters
    ----------
    topics - (n_topics, top_words) array
    corpus - 2d array - tokenized sequences of words in the corpus

    Returns
    -------
    NPMI scores per topic
    """
    K = topics.shape[0]  # number of topics
    T = topics.shape[1]  # top words in a topic
    word_frequencies, joint_word_frequencies = calculate_word_frequencies(topics, corpus)
    # dictionaries; word_frequencies={w_i : p_wi}, joint_word_frequencies={(w_i, w_j):p_wi_wj}

    topic_coherences = []
    for i in range(K):
        topic_coherence = 0
        for j in range(T):
            # npmi for each word in a topic
            word_npmi = 0
            w_j = topics[i][j]
            for k in range(T):
                if k == j: continue
                w_k = topics[i][k]
                p_wjwk = word_frequencies[(w_j, w_k)]
                if (p_wjwk-0) < 1e-12:
                    # the joint frequency is zero, skip
                    continue
                p_wj = word_frequencies[w_j]
                p_wk = word_frequencies[w_k]
                word_npmi += log(p_wjwk/(p_wj*p_wk)) / (-log(p_wjwk))

            topic_coherence += word_npmi

        topic_coherence = topic_coherence / T
        topic_coherences.append(topic_coherence)

    return topic_coherences
