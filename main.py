# TODO - customize runner and trainers - extend to new classes
#  TODO - load all parameters from configs - check out
import argparse
import os
import time
from multiprocessing import cpu_count

from tensorboardX import SummaryWriter
import torch.utils.data
from torch import optim

from datasets import DatasetLoader
from models.architectures.decoders.fc import FCDecoder
from models.architectures.encoders.fc import FCEncoder
from models.vae import VAE
from preprocess.vectorizer import Vectorizer
from trainer import VAETrainer
from utils.nn_utils import load_embeddings


# TODO - needs update
def main(args):
    args.cuda = args.cuda and torch.cuda.is_available()
    if torch.cuda.is_available():
        print('Cuda will be used for training.')

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': cpu_count(), 'pin_memory': True} if args.cuda else {'num_workers': cpu_count()}

    X_train, X_test = DatasetLoader().load_dateset('20newsgroups')
    X_train_texts, X_test_texts = [d['text'] for d in X_train], [d['text'] for d in X_test]
    vectorizer = Vectorizer(max_sequence_length=15, min_occ=5)
    vectorizer.load_vocab('resources/datasets/20_newsgroups_old/vocab_full.json')
    vectorizer.extract_embeddings(load_embeddings('resources/embeddings/googlenews-vectors-300.bin'))
    X_train_vec, X_test_vec = vectorizer.text_to_embeddings(X_train_texts, maxlen_ratio=0.9), \
                              vectorizer.text_to_embeddings(X_test_texts, maxlen_ratio=0.9)
    train_loader = torch.utils.data.DataLoader(dataset=X_train_vec, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=X_test_vec, batch_size=args.batch_size, shuffle=True, **kwargs)

    model = VAE(encoder=FCEncoder(input_dim=300, latent_dim=20),
                decoder=FCDecoder(latent_dim=20, output_dim=300),
                recon_distribution='gauss').to(device)
    model.double()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)
    writer = None
    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.log_dir, ts))
        writer.add_text("model_0", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    VAETrainer(model, device, train_loader, test_loader, save_model_path, writer).run(optimizer, args.epochs,
                                                                                      sample_every=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE text runner')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: a_0)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model_path', type=str, default='bin')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('tensorboard_logging', action='store_true')
    args = parser.parse_args()

    main(args)
