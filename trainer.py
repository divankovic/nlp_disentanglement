import torch
import os
import time


class VAETrainer:

    def __init__(self, model, device, train_loader, test_loader, included_labels=False, save_model_path=None,
                 writer=None, log_interval=10,
                 test_epoch=1, probtorch=False):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.included_labels = included_labels
        self.log_interval = log_interval
        self.test_epoch = test_epoch
        self.save_model_path = save_model_path
        self.writer = writer
        self.probtorch = probtorch

    def run(self, optimizer, epochs):
        for epoch in range(1, epochs + 1):
            self.train(epoch, optimizer)
            if epoch % self.test_epoch == 0:
                self.test(epoch)

        if self.save_model_path:
            if not os.path.exists(self.save_model_path):
                os.makedirs(self.save_model_path)
            # save checkpoint
            torch.save(self.model.state_dict(), os.path.join(self.save_model_path, 'model.pt'))
            print("Model saved at %s" % self.save_model_path)

    def train(self, epoch, optimizer):
        self.model.train()
        train_loss = 0
        batch_size = self.train_loader.batch_size
        for batch_idx, data in enumerate(self.train_loader):
            if self.included_labels:
                data, labels = data

            data = data.to(self.device)
            optimizer.zero_grad()

            if self.probtorch:
                if len(data) != batch_size:
                    # hfvae needs complete batches to work
                    continue
                data = data.cuda()
                q, p = self.model(data)
                loss = self.model.loss_function(q, p, len(self.train_loader.dataset), len(data))
            else:
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
        if self.writer:
            self.writer.add_scalar('train loss', train_loss / len(self.train_loader.dataset), epoch)

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            batch_size = self.train_loader.batch_size
            for i, data in enumerate(self.test_loader):
                if self.included_labels:
                    data, labels = data

                data = data.to(self.device)
                if self.probtorch:
                    if len(data) != batch_size:
                        # hfvae needs complete batches to work
                        continue
                    data = data.cuda()
                    q, p = self.model(data)
                    test_loss += self.model.loss_function(q, p, len(self.test_loader.dataset), len(data)).item()
                else:
                    recon_batch, mu, logvar = self.model(data)
                    test_loss += self.model.loss_function(recon_batch, data, mu, logvar).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        if self.writer:
            self.writer.add_scalar('test loss', test_loss, epoch)
