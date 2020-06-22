import argparse
import os
import sys
import matplotlib.pyplot as plt

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
from evaluation import evaluate_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glob_dir', '-c', dest='glob_dir',
                        help='path to the directory containing directories with the trained models and configs. The '
                             'results are saved to that same directory by default',
                        default='results/grid/')
    args = parser.parse_args()

    print('Starting evaluation for models in directory %s' % args.glob_dir)

    models = os.listdir(args.glob_dir)
    for (i, model) in enumerate(models):
        print('Running evaluation for model %s,  %d/%d' % (model, i + 1, len(models)))
        evaluate_model.main(
            ['--model_path', os.path.join(args.glob_dir, model), '--save_path', os.path.join(args.glob_dir, model)])

    print('Evaluation complete!')
