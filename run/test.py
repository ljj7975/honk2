import argparse
import os
import random
from datetime import datetime
from pprint import pprint

import torch
import numpy as np
import torch.optim as optimizer_modules
from tqdm import tqdm

from .run_utils import merge_configs, init_data_loader, set_seed
from dataset import DatasetType
from metric import collect_metrics
from utils import Workspace, find_cls, load_json, prepare_device


def evaluate(device, prefix, model, data_loader, loss_fn, metrics, label_mapping):
    total_loss = 0

    model.eval()
    for _, (data, target) in enumerate(tqdm(data_loader, desc=f"Evaluating {prefix} dataset")):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)

        loss = loss_fn(output, target)

        total_loss += loss.item()

        for metric in metrics.values():
            metric.accumulate(output, target)

    results = {
        "loss": total_loss / len(data_loader)
    }

    results.update(collect_metrics(metrics, label_mapping))

    return results

def main(config):
    set_seed(config["seed"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # prepare hardware accelearation
    device, gpu_device_ids = prepare_device(config["num_gpu"])


    # Preapre model
    n_labels = len(config["target_class"])
    if config["unknown_class"]:
        n_labels += 1
    if config["silence_class"]:
        n_labels += 1

    model_config = config["model"]
    model_class = find_cls(f"model.{model_config['name']}")

    model_config["config"]["n_labels"] = n_labels
    model = model_class(model_config["config"])
    model_name = type(model).__name__


    # Multiple GPU supports
    if len(gpu_device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_device_ids)

    model.to(device)
    print("model:\n", model)

    test_data_loader = init_data_loader(config, DatasetType.TEST)
    print(f"test dataset size: {len(test_data_loader.dataset)}")

    label_mapping = test_data_loader.dataset.label_mapping


    # Model evaluation initialization
    loss_fn = find_cls(f"loss_fn.{config['loss_fn']}")

    metrics = {}
    for metric_name in config['metric']:
        metrics[metric_name] = find_cls(f"metric.{metric_name}")

    trained_model_dir = config["evaluate_model_dir"]
    workspace = Workspace(device, trained_model_dir, model, loss_fn, metrics)


    # load the model to evaluate
    if "evaluate_epoch" in config:
        checkpoint = workspace.load_checkpoint(config["evaluate_epoch"])
    else:
        checkpoint = workspace.load_best_model()

    print("Training results")
    print("\tbest_epoch: {}".format(checkpoint["best_epoch"]))
    print("\tbest_dev_loss: {}".format(checkpoint["best_dev_loss"]))
    print("\tbest_dev_metric: {}".format(checkpoint["best_dev_metric"]))


    # Test model
    test_results = evaluate(device, "test", model, test_data_loader, loss_fn, metrics)

    print("Test results")
    pprint(test_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")

    parser.add_argument("--config", default=None, required=True, type=str,
                        help="path to config file")

    args = parser.parse_args()

    main(load_json(args.config))
