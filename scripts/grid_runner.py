import argparse
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
import run

# TODO - add parallelization - run multiple experiments at the same time
# utilize this only if there is memory on the gpu(s), multiple gpus, ...
# for now simple
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', '-c', dest='configs',
                        help='path to the directory containing config files for the experiments which should be run',
                        default='configs/20news/')
    args = parser.parse_args()

    print('Starting experiment grid from %s' % args.configs)
    configs = os.listdir(args.configs)
    for (i, config) in enumerate(configs):
        print('Running experiment %d/%d' % (i+1, len(configs)))
        run.main(['--config', args.configs+config])

    print('Experiments complete!')