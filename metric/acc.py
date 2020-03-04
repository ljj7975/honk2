import torch

from .metric_utils import MacroMetric
from utils import register_cls


@register_cls('metric.Acc')
class Acc(MicroMetric):
    def __init__(self):
        super().__init__()
        self.total = 0
        self.correct = 0

    def accumulate(self, output, target):
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = torch.sum(pred == target).item()
            total = len(target)

        self.correct += correct
        self.total += total

        return correct/total

    def get_metric(self):
        return self.correct / self.total

    def reset_metric(self):
        self.correct = 0
        self.total = 0
