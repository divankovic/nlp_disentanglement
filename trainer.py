import torch
from torchvision.utils import save_image


class VAETrainer:
    # TODO - add model saving
    # TODO - customize experiments - load all parameters from configs

    def __init__(self, model, device, train_loader, test_loader, log_interval=10, test_epoch=1):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.log_interval = log_interval
        self.test_epoch = test_epoch

    def run(self, optimizer, epochs, sample_every=0):
        for epoch in range(1, epochs + 1):
            self.train(epoch, optimizer)
            if epoch % self.test_epoch == 0:
                self.test(epoch)

            if sample_every != 0 and epoch % sample_every == 0:
                # sample some instances from the VAE model and save them
                # for mnist - TODO - customize this
                with torch.no_grad():
                    sample = torch.randn(64, 20).to(self.device)
                    sample = self.model.decode(sample).cpu()
                    save_image(sample.view(64, 1, 28, 28),
                               'results/mnist/sample_' + str(epoch) + '.png')

    def train(self, epoch, optimizer):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: %2d [%6d/%6d (%3.0f%%)]\tLoss: %.6f' % (
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.model.loss_function(recon_batch, data, mu, logvar).item()
                if i == 0:
                    # TODO - also customize this
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                            recon_batch.view(self.test_loader.batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                               'results/mnist/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
