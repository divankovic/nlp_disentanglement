import os
import torch
import probtorch
from utils.file_handling import load_nparray
from models.architectures.encoders.fc import HFVAEFCEncoder
from models.architectures.decoders.fc import HFVAEFCDecoder
from models.hfvae import HFVAE
from multiprocessing import cpu_count
from trainer import VAETrainer
import time

os.chdir('..')
print(os.getcwd())
# model parameters
HIDDEN_DIM = 256
LATENT_DIM = 50

# training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
BETA1 = 0.90
EPS = 1e-9
CUDA = torch.cuda.is_available()

# saving
SAVE_PATH = 'results/hfvae'


kwargs = {'num_workers': cpu_count(), 'pin_memory': True} if CUDA else {'num_workers': cpu_count()}
train_loader = torch.utils.data.DataLoader(load_nparray('resources/datasets/20_newsgroups/preprocessed/train_bow.txt'),
                                         batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(load_nparray('resources/datasets/20_newsgroups/preprocessed/test_bow.txt'),
                                        batch_size=BATCH_SIZE, shuffle=True, **kwargs)


enc = HFVAEFCEncoder(input_dim=2000, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
dec = HFVAEFCDecoder(latent_dim=LATENT_DIM, output_dim=2000, batch_size=BATCH_SIZE)
model = HFVAE(encoder=enc, decoder=dec)
model.cuda()
model.double()
optimizer =  torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
device = torch.device("cuda" if CUDA else "cpu")

VAETrainer(model, device, train_loader, test_loader, save_model_path=SAVE_PATH, probtorch=True).run(optimizer, NUM_EPOCHS)
