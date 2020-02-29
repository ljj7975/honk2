import argparse
import os
import random
from datetime import datetime

import torch
import numpy as np
import torch.optim as optimizer_modules
from tqdm import tqdm

from .runnable_utils import merge_configs, init_data_loader, set_seed
from utils import DatasetType, Workspace
from utils import find_cls, load_json, prepare_device

def evaluate(device, prefix, model, data_loader, loss_fn, metric):
    total_loss = 0
    total_metric = 0

    model.eval()
    for _, (data, target) in enumerate(tqdm(data_loader, desc=f"Evaluating {prefix} dataset")):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)

        loss = loss_fn(output, target)

        total_loss += loss.item()
        total_metric += metric(output, target)

    log = {
        "loss": total_loss / len(data_loader),
        "metric": total_metric / len(data_loader)
    }

    return log

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
    model_class = find_cls(f"model.{model_config['name'].lower()}")

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


    # Model evaluation initialization
    loss_fn = find_cls(f"loss_fn.{config['loss_fn'].lower()}")

    # TODO:: add flexibility for best metric (i.e. max, min)
    metric = find_cls(f"metric.{config['metric'].lower()}")

    trained_model_dir = config["evaluate_model_dir"]
    workspace = Workspace(device, trained_model_dir, model, loss_fn, metric)


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
    test_log = evaluate(device, "test", model, test_data_loader, loss_fn, metric)

    print("Test results")
    print("\ttest_loss: {}".format(test_log["loss"]))
    print("\ttest_metric: {}".format(test_log["metric"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")

    parser.add_argument("--config", default=None, required=True, type=str,
                        help="path to config file")

    args = parser.parse_args()

    main(load_json(args.config))
