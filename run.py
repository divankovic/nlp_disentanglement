from utils.nn_utils import set_seed
import argparse
import ruamel.yaml as yaml
from models import *
from experiment import *
import os
import sys
import time
from utils.file_handling import MultiOutput
from trainer import VAETrainer


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        dest="config",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/basic/hfvae.yaml')
    args = parser.parse_args(args)
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # set seed for reproducibility
    set_seed(config['experiment_parameters']['seed'])
    # save used parameters to save directory
    SAVE_PATH = config['experiment_parameters']['save_path']
    ts = time.strftime('%Y-%m-%d-%H:%M', time.gmtime())
    save_path = os.path.join(SAVE_PATH, config['experiment_parameters']['dataset'],
                             config['model_parameters']['name'], ts)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    model_params = {
        'encoder': encoders[config['model_parameters']['encoder']['name']](**config['model_parameters']['encoder']),
        'decoder': decoders[config['model_parameters']['decoder']['name']](**config['model_parameters']['decoder']),
    }
    if 'spec' in config['model_parameters']:
        model_params.update(config['model_parameters']['spec'])

    model = vae_models[config['model_parameters']['name']](**model_params)
    experiment = experiments[config['experiment_parameters']['name']](model, config['experiment_parameters'])

    logger = MultiOutput(sys.stdout, open(os.path.join(save_path, 'training_output.txt'), 'w'))
    print('======== Running experiment %s ========' % args.config)
    VAETrainer(experiment, logger, save_path, config['trainer_parameters']).run()


if __name__ == '__main__':
    main(sys.argv[1:])
