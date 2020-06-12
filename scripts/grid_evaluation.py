import argparse
import os
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
        print('Running evaluation for model %s,  %d/%d' % (model, i+1, len(models)))
        evaluate_model.main(['--model_path', args.glob_dir+model, '--save_path', args.glob_dir+model])

    print('Evaluation complete!')