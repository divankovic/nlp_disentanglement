import argparse
import ruamel.yaml as yaml
import os
import sys
import time
from utils.file_handling import MultiOutput, load_model_from_config
from evaluation.vizualizations import tsne_plot, correlation_plot


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
                        help='path to which the save the results to',
                        default='results/20news/HFVAE/2020-06-11-17:02')
    args = parser.parse_args(args)
    config_path = os.path.join(args.model_path, 'config.yaml')
    config = yaml.safe_load(open(config_path, 'r'))
    data_path = config['experiment_parameters']['data_path']
    model = load_model_from_config(config_path, weights_path=os.path.join(args.model_path, 'model.pt'))
    model.eval()
    experiment = experiments[config['experiment_parameters']['name']](model, config['experiment_parameters'])

    # start evaluation methods
    zs, z_mus = experiment.sample_latent(experiment.test_dataloader())
    ys = np.load(os.path.join(data_path, 'test.labels.npy'))
    labels = json.load(open(os.path.join(data_path, 'labels.json'), 'r'))
    tsne_plot(zs, ys, labels, show=False, save_path=args.save_path, plot_by_class=True,
              perplexity=10, learning_rate=200, n_iter=2000, n_jobs=-1)
    # consider early_exaggeration and init (pca)

    #correlation plot
    #topics
    #npmi
    #...


if __name__ == '__main__':
    main(sys.argv[1:])
