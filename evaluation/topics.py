from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import Dense2Corpus
import numpy as np


# TODO : test
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


# TODO : test
def coherence_score(model, X, idx2word, score_num=7):
    """
    Computes the coherence score of the learned topics using gensim's implementation.
    Parameters
    ----------
    model - trainedmodel
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
    id2word = {index: str(index) for index in range(X.shape[1])}
    topics = [
        [str(item.item()) for item in topic]
        for topic in decoder_weight.topk(min(score_num, X.shape[1]), dim=0)[1].t()
    ]

    coherence_model = CoherenceModel(topics=topics, corpus=corpus, dictionary=Dictionary.from_corpus(corpus, id2word),
                                     coherence='u_mass')
    # return coherence_model.get_coherence()
    return coherence_model.get_coherence_per_topic()
