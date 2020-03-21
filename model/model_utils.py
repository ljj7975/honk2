import torch.nn as nn

from abc import ABC, abstractmethod


class BaseModel(ABC, nn.Module):
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # @abstractmethod
    # def size(self):
    #     # TODO:: approximated required memory
    #     pass
