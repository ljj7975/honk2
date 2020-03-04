import torch

from .metric_utils import MicroMetric
from utils import register_cls


@register_cls('metric.PerClassAcc')
class PerClassAcc(MacroMetric):
    def __init__(self):
        super().__init__()
        self.correct = {}
        self.total = {}

    def accumulate(self, output, target):
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)

            pred = pred.data.tolist()
            target = target.data.tolist()

        total = {}
        correct = {}

        for guess, ground_truth in zip(pred, target):
            if ground_truth not in total:
                total[ground_truth] = 0
                correct[ground_truth] = 0

            total[ground_truth] += 1
            if guess == ground_truth:
                correct[ground_truth] += 1

        acc = {}
        for key, val in total.items():
            acc[key] = correct[key] / val

            if key not in self.total:
                self.total[key] = 0
                self.correct[key] = 0

            self.total[key] += val
            self.correct[key] += correct[key]

        return acc

    def get_metric(self):
        acc = {}
        for key, val in self.total.items():
            acc[key] = self.correct[key] / val
        return acc

    def reset_metric(self):
        self.correct = {}
        self.total = {}
