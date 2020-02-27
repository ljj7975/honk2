import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from .file_utils import ensure_dir


class Workspace():
    def __init__(self, device, dir, model, loss_fn, metric, optimizer=None):
        self.device = device
        self.dir = dir
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.optimizer = optimizer

        path = Path(self.dir)
        if not path.exists():
            # Training case
            ensure_dir(self.dir)

        log_path = path / 'logs'
        self.summary_writer = SummaryWriter(str(log_path))

    def _get_checkpoint_path(self, epoch):
        return Path(self.dir, "checkpoint_{}.pt".format(epoch))

    def _get_best_model_path(self):
        return Path(self.dir, "best_model.pt")

    def _save(self, path, accessories):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'loss_fn': self.loss_fn,
            'metric': self.metric,
            'optimizer_state_dict': self.optimizer.state_dict()
            }

        checkpoint.update(accessories)

        torch.save(checkpoint, path)

    def save_checkpoint(self, epoch, accessories):
        accessories['epoch'] = epoch
        path = self._get_checkpoint_path(epoch)

        if not os.path.exists(path):
            self._save(path, accessories)

    def save_best_model(self, accessories):
        path = self._get_best_model_path()
        self._save(path, accessories)

    def _load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint.pop('model_state_dict'))
        self.loss_fn = checkpoint.pop('loss_fn')
        self.metric = checkpoint.pop('metric')

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint.pop('optimizer_state_dict'))

        return checkpoint

    def load_checkpoint(self, epoch):
        path = self._get_checkpoint_path(epoch)
        print(f"loading {path}")
        return self._load(path)

    def load_best_model(self):
        path = self._get_best_model_path()
        print(f"loading {path}")
        return self._load(path)
