import torch
import os


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

    def train(self, epoch, optimizer, track_mutual_info=True):
        self.model.train()
        batch_losses = []
        batch_size = self.train_loader.batch_size
        if track_mutual_info:
            batch_mis = []
        for batch_idx, data in enumerate(self.train_loader):
            if self.included_labels:
                data, labels = data

            data = data.to(self.device).cuda().double()
            optimizer.zero_grad()

            if self.probtorch:
                if len(data) != batch_size:
                    # probtorch implementation needs complete batches to work
                    continue
                q, p = self.model(data)
                loss = self.model.loss_function(q, p, N=len(self.train_loader.dataset), batch_size=len(data))
                # the loss is already normalized, no need to further divide it
                loss_item = loss.item()

                # extra for HFVAE - mutual information
                batch_mis.append(
                    self.model.mutual_info(q, p, N=len(self.train_loader.dataset), batch_size=len(data)).item())
            else:
                recon_batch, mu, logvar = self.model(data)
                loss = self.model.loss_function(recon_batch, data, mu, logvar)
                loss_item = loss.item() / len(data)  # normalize by batch

            loss.backward()
            batch_loss = loss_item
            batch_losses.append(batch_loss)
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: %2d [%6d/%6d (%3.0f%%)]\tLoss: %.6f' % (
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    batch_loss))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, sum(batch_losses) / len(batch_losses)))
        if track_mutual_info:
            print('====>         Average I(x,z): {:.4f}'.format(sum(batch_mis) / len(batch_mis)))

        if self.writer:
            self.writer.add_scalar('train loss', sum(batch_losses) / len(batch_losses), epoch)

    def test(self, epoch):
        self.model.eval()
        batch_losses = []
        with torch.no_grad():
            batch_size = self.train_loader.batch_size
            for i, data in enumerate(self.test_loader):
                if self.included_labels:
                    data, labels = data

                data = data.to(self.device).cuda().double()

                if self.probtorch:
                    if len(data) != batch_size:
                        # hfvae needs complete batches to work
                        continue
                    q, p = self.model(data)
                    batch_losses.append(
                        self.model.loss_function(q, p, N=len(self.test_loader.dataset), batch_size=len(data)).item())
                else:
                    recon_batch, mu, logvar = self.model(data)
                    batch_losses.append(self.model.loss_function(recon_batch, data, mu, logvar).item() / len(data))

        test_loss = sum(batch_losses) / len(batch_losses)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        if self.writer:
            self.writer.add_scalar('test loss', test_loss, epoch)
