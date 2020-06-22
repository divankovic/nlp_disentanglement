import argparse
import os
import sys
import pickle

import ruamel.yaml as yaml

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
import json
from utils.file_handling import load_model_from_config
from utils.nn_utils import set_seed
from evaluation.vizualizations import correlation_plot
from evaluation.topics import get_topics, npmi_coherence_score, print_top_words, get_most_correlated_topics
from experiment import *


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m',
                        dest="model_path",
                        metavar='FILE',
                        help='path to the trained model',
                        default='results/20news/HFVAE/2020-06-11-17:02')
    parser.add_argument('--save_path', '-s',
                        dest="save_path",
                        metavar='FILE',
                        help='path to which the save the results to, if empty defaults to the model path',
                        default='')
    parser.add_argument('--most_cor', '-c',
                        dest="most_correlated",
                        action='store_false')

    args = parser.parse_args(args)
    if args.save_path:
        SAVE_PATH = args.save_path
    else:
        SAVE_PATH = args.model_path
    config_path = os.path.join(args.model_path, 'config.yaml')
    config = yaml.safe_load(open(config_path, 'r'))
    data_path = config['experiment_parameters']['data_path']
    model = load_model_from_config(config_path, weights_path=os.path.join(args.model_path, 'model.pt'))
    model.eval()
    experiment = experiments[config['experiment_parameters']['name']](model, config['experiment_parameters'])
    set_seed(config['experiment_parameters']['seed'])

    # start evaluation methods
    zs, z_mus = experiment.sample_latent(experiment.test_dataloader())
    ys = np.load(os.path.join(data_path, 'test.labels.npy'))[:zs.shape[0]]
    labels = json.load(open(os.path.join(data_path, 'labels.json'), 'r'))
    # tsne_plot(zs, ys, labels, show=False, save_path=SAVE_PATH, plot_by_class=True,
    #          perplexity=10, learning_rate=200, n_iter=2000, n_jobs=-1)
    # consider early_exaggeration and init (pca)

    # correlation plot
    correlation_plot(zs, show=False, save_path=SAVE_PATH)

    # topics
    n_top = 10
    vocab = json.load(open(os.path.join(data_path, 'vocab.json'), 'r'))
    idx2word = {i: vocab[i] for i in range(len(vocab))}
    beta = model.decoder.main[0].weight.cpu().detach().numpy().T
    topics = get_topics(beta, idx2word, n_top=n_top)
    print_top_words(beta, idx2word, n_top=n_top, save_path=os.path.join(SAVE_PATH, 'topics.txt'))

    # npmi
    if not os.path.exists(os.path.join(data_path, 'word_frequencies.pkl')) or not os.path.exists(
            os.path.join(data_path, 'jointword_frequencies.pkl')):
        raise ValueError(
            'word_frequencies.pkl or jointword_frequencies.pkl not found in data path! Run '
            'calculate_word_frequencies() from topics.py on the desired dataset (X_raw.json) to obtain it.')
    word_frequencies = pickle.load(open(os.path.join(data_path, 'word_frequencies.pkl'), 'rb'))
    joint_word_frequencies = pickle.load(open(os.path.join(data_path, 'jointword_frequencies.pkl'), 'rb'))
    npmi_per_topic = npmi_coherence_score(topics, word_frequencies, joint_word_frequencies)
    print(npmi_per_topic)
    print(sum(npmi_per_topic) / len(npmi_per_topic))
    print('Max: %f' % max(npmi_per_topic))
    print('Topic : %s' % (' '.join(topics[np.argmax(npmi_per_topic)])))
    # save results
    json.dump({'npmi_per_topic': npmi_per_topic, 'avg_npmi': sum(npmi_per_topic) / len(npmi_per_topic),
               'topics': topics},
              open(os.path.join(SAVE_PATH, 'npmi.json'), 'w'), indent=4)

    # npmi for most correlated topics
    if args.most_correlated:
        inds, covs, cor_topics = get_most_correlated_topics(np.corrcoef(zs.T), top_correlations=4)
        for topic in cor_topics:
            print('%d %f %s' % (topic, npmi_per_topic[topic], topics[topic]))


if __name__ == '__main__':
    main(sys.argv[1:])
