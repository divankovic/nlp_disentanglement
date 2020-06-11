import torch
from multiprocessing import cpu_count
import numpy as np
import os


class VAEXperiment:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.cuda = params['use_cuda'] and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        if self.cuda:
            self.model = model.cuda().double()

        self.loader_kwargs = {'num_workers': cpu_count(), 'pin_memory': True} if self.cuda else {
            'num_workers': cpu_count()}

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def train_step(self, batch, **kwargs):
        return self.__step(batch, **kwargs)

    def test_step(self, batch, **kwargs):
        return self.__step(batch, **kwargs)

    def __step(self, batch, **kwargs):
        batch = batch.to(self.device)
        if self.cuda: batch = batch.cuda().double()
        recon_batch, mu, logvar = self.model(batch)
        loss = self.model.loss_function(recon_batch, batch, mu, logvar)
        ret = {'loss': loss}
        if kwargs['loss_components']:
            loss_components = self.model.loss_components(recon_batch, batch, mu, logvar)
            for (k, v) in loss_components.items():
                ret[k] = v.item()

        return ret

    def epoch_end(self, metrics):
        metrics = {k: np.mean(v) for k, v in metrics.items()}
        return metrics

    def configure_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])

    def train_dataloader(self):
        return self.__data_loader(split='train')

    def test_dataloader(self):
        return self.__data_loader(split='test')

    def __data_loader(self, split='train'):
        if self.params['sparse']:
            from scipy.sparse import load_npz
            data = load_npz(os.path.join(self.params['data_path'], split + '.npz')).toarray()
        else:
            data = np.load(os.path.join(self.params['data_path'], split + '.npy'))

        if self.params['labels']:
            labels = np.load(os.path.join(self.params['data_path'], split + '.labels.npy'))
            return torch.utils.data.DataLoader([[x, y] for x, y in zip(data, labels)],
                                               batch_size=self.params['batch_size'], shuffle=True, **self.loader_kwargs)

        return torch.utils.data.DataLoader(data, batch_size=self.params['batch_size'], shuffle=True,
                                           **self.loader_kwargs)


class PTVAEXperiment(VAEXperiment):
    def __init__(self, model, params):
        super().__init__(model, params)
        # drop last incomplete batch for probtorch
        self.loader_kwargs['drop_last'] = True

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def train_step(self, batch, **kwargs):
        return self.__step(batch, split='train', **kwargs)

    def test_step(self, batch, **kwargs):
        return self.__step(batch, split='test', **kwargs)

    def __step(self, batch, split='train', **kwargs):
        if split == 'train':
            N = self.train_len
        else:
            N = self.test_len

        batch = batch.to(self.device)
        if self.cuda: batch = batch.cuda().double()
        q, p = self.model(batch)
        loss = self.model.loss_function(q, p, N=N, batch_size=self.params['batch_size'])
        ret = {'loss': loss}
        if kwargs['loss_components']:
            loss_components = self.model.loss_components(q, p, N=N, batch_size=self.params['batch_size'])
            for (k, v) in loss_components.items():
                ret[k] = v.item()

        if kwargs['perplexity']:
            elbos = - self.model.loss_function(q, p, N=N, batch_size=self.params['batch_size'],
                                               reduce=False)
            N_d = batch.sum(-1)  # length of individual documents in the batch
            ret['perplexity'] = torch.mul(elbos, (1 / N_d)).sum().item()

        return ret

    def epoch_end(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if k == 'perplexity':
                new_metrics[k] = np.exp(-1 / (len(v) * self.params['batch_size']) * np.sum(v))
            else:
                new_metrics[k] = np.mean(v)

        return new_metrics

    def train_dataloader(self):
        loader = super().train_dataloader()
        # needed for bias in probtorch models
        self.train_len = len(loader)
        return loader

    def test_dataloader(self):
        loader = super().test_dataloader()
        # needed for bias in probtorch models
        self.test_len = len(loader)
        return loader


experiments = {'VAEXperiment': VAEXperiment,
               'PTVAEXperiment': PTVAEXperiment}
