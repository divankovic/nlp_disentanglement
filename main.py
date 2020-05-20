# TODO - customize runner and trainers - extend to new classes
import argparse
import torch.utils.data
from torch import optim
from models.architectures.encoders.fc import FCEncoder
from models.architectures.decoders.fc import FCDecoder

from torchvision import datasets, transforms
from models.vae import VAE
from trainer import VAETrainer

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if torch.cuda.is_available():
    print('Cuda is available!')

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

model = VAE(encoder=FCEncoder(input_dim=28 * 28, latent_dim=20),
            decoder=FCDecoder(latent_dim=20, output_dim=784)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    VAETrainer(model, device, train_loader, test_loader).run(optimizer, args.epochs, sample_every=1)
