import json
import os
from collections import defaultdict

import torch

from utils.torch_utils import EarlyStopping


class VAETrainer:

    def __init__(self, experiment, logger, save_path, params):
        self.experiment = experiment
        self.logger = logger
        self.save_path = save_path
        self.params = params

    def run(self):
        model_save_path = os.path.join(self.save_path, 'model.pt')
        optimizer = self.experiment.configure_optimizer()
        metrics = {'train': defaultdict(list), 'test': defaultdict(list)}
        early_stopping = None
        if self.params['patience'] > 0:
            early_stopping = EarlyStopping(patience=self.params['patience'])

        for epoch in range(1, self.params['max_epochs'] + 1):
            train_metrics = self.train(epoch, optimizer)
            for k, v in train_metrics.items(): metrics['train'][k].append(v)

            if epoch % self.params['test_interval'] == 0:
                test_metrics = self.test()
                for k, v in test_metrics.items(): metrics['test'][k].append(v)
                if early_stopping:
                    if early_stopping(test_metrics['loss']):
                        self.logger.print('Patience limit reached. Stopping training...')
                        break
                    else:
                        torch.save(self.experiment.model.state_dict(), model_save_path)
                        self.logger.print('Model saved at %s' % model_save_path)

            self.logger.print('')

        if early_stopping is None:
            torch.save(self.experiment.model.state_dict(), model_save_path)
            self.logger.print('Model saved at %s' % model_save_path)
        json.dump(metrics, open(os.path.join(self.save_path, 'metrics.json'), 'w'), indent=4)
        self.logger.print('Metrics saved at %s'%os.path.join(self.save_path, 'metrics.json'))
        self.logger.print('\n')
        return metrics

    def train(self, epoch, optimizer):
        self.experiment.model.train()
        metrics = defaultdict(list)
        train_loader = self.experiment.train_dataloader()
        for batch_idx, data in enumerate(train_loader):
            if self.experiment.params['labels']: (data, _) = data
            optimizer.zero_grad()
            step_metrics = self.experiment.train_step(data, **self.params['track'])
            loss = step_metrics['loss']
            loss_item = loss.item()
            step_metrics['loss'] = loss_item
            for metric in step_metrics:
                metrics[metric].append(step_metrics[metric])
            loss.backward()
            optimizer.step()

            if batch_idx % self.params['log_interval'] == 0:
                self.logger.print('Train Epoch: %2d [%6d/%6d (%3.0f%%)]\tLoss: %.6f' % (
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss_item))

        metrics = self.experiment.epoch_end(metrics)
        for k, v in metrics.items():
            self.logger.print('====>    Average %s: %.4f' % (k, v))

        return metrics

    def test(self):
        self.experiment.model.eval()
        metrics = defaultdict(list)
        with torch.no_grad():
            test_loader = self.experiment.test_dataloader()
            for batch_idx, data in enumerate(test_loader):
                if self.experiment.params['labels']: (data, _) = data
                step_metrics = self.experiment.test_step(data, **self.params['track'])
                step_metrics['loss'] = step_metrics['loss'].item()
                for metric in step_metrics:
                    metrics[metric].append(step_metrics[metric])

        metrics = self.experiment.epoch_end(metrics)
        self.logger.print('Test metrics ')
        for k, v in metrics.items():
            self.logger.print('====>    Average %s: %.4f' % (k, v))

        return metrics