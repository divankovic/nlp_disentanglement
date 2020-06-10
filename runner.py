from utils.nn_utils import set_seed
import argparse
import yaml
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c',
                    dest="config",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/hfvae.yaml')

args = parser.parse_args()
with open(args.config, 'r') as file:
    config = yaml.load(file)


# TODO - also save the used yml file to output, to avoid later confusion
set_seed(config['experiment_params']['seed'])
# load model
# create 'experiment'
# call trainer for training
# use lightning or do it all yourself +++ :)
