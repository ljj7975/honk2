from abc import ABC, abstractmethod
from enum import Enum


class MetricType(Enum):
    MACRO = "MACRO"
    MICRO = "MICRO"

class Metric(ABC):
    @abstractmethod
    def accumulate(self, output, target):
        pass

    @abstractmethod
    def get_metric(self):
        pass

    @abstractmethod
    def reset_metric(self):
        pass


class MicroMetric(Metric):
    def __init__(self):
        self.type = MetricType.MACRO

    def get_type(self):
        return self.type


class MacroMetric(Metric):
    def __init__(self):
        self.type = MetricType.MICRO

    def get_type(self):
        return self.type


def collect_metrics(metrics, label_mapping):
    results = {}
    for metric_name, metric in metrics.items():
        if metric.get_type() == MetricType.MACRO:
            results[f"metric_{metric_name}"] = metric.get_metric()
        elif metric.get_type() == MetricType.MICRO:
            micro_metric = metric.get_metric()
            relabelled = {}
            for key, val in micro_metric.items():
                label = label_mapping[key]
                relabelled[label] = val
            results[f"metric_{metric_name}"] = relabelled

    return results
