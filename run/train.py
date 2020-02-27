import argparse
import copy
import os
import random
from datetime import datetime

import torch
import numpy as np
import torch.optim as optimizer_modules
from tqdm import tqdm

from utils import DatasetType, Workspace
from utils import find_cls, load_json, prepare_device


def merge_configs(base_config, additional_config):
    new_config = copy.deepcopy(base_config)
    for key, value in additional_config.items():
        new_config[key] = value

    return new_config

def init_data_loader(config, type):
    type_key = type.value
    dataset_name = config["datasets"][type_key]["dataset"]["name"]
    dataset_config = config["datasets"][type_key]["dataset"]["config"]
    dataset_config = merge_configs(config[dataset_name], dataset_config)
    dataset_config["target_class"] = config["target_class"]
    dataset_config["unknown_class"] = config["unknown_class"]
    dataset_config["silence_class"] = config["silence_class"]
    dataset_config["type"] = type

    data_loader_name = config["datasets"][type_key]["data_loader"]["name"]
    data_loader_config = config["datasets"][type_key]["data_loader"]["config"]
    data_loader_config = merge_configs(config[data_loader_name], data_loader_config)

    data_loader_class = find_cls(f"data_loader.{data_loader_name.lower()}")
    data_loader = data_loader_class(data_loader_config, dataset_config)

    return data_loader

def main(config):

    # set random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])


    # prepare hardware accelearation
    device, gpu_device_ids = prepare_device(config["num_gpu"])

    if "cuda" in str(device):
        print(f"utilizing gpu devices : {gpu_device_ids}")
        torch.cuda.manual_seed(config["seed"])


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

    if "trained_model" in model_config:
        trained_model = model_config["trained_model"]
        print(f"pretrained model: {trained_model}")

        model.load_state_dict(torch.load(trained_model, map_location=device), strict=False)


    # Multiple GPU supports
    if len(gpu_device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_device_ids)

    model.to(device)
    print("model:\n", model)


    # Prepare DataLoader
    train_data_loader = init_data_loader(config, DatasetType.TRAIN)
    print(f"training dataset size: {len(train_data_loader.dataset)}")

    dev_data_loader = init_data_loader(config, DatasetType.DEV)
    print(f"dev dataset size: {len(dev_data_loader.dataset)}")

    test_data_loader = init_data_loader(config, DatasetType.TEST)
    print(f"test dataset size: {len(test_data_loader.dataset)}")


    # Optimizer
    optimizer_config = config["optimizer"]
    optimizer_config["config"]["params"] = model.parameters()
    optimizer_class = getattr(optimizer_modules, optimizer_config["name"])
    optimizer = optimizer_class(**optimizer_config["config"])

    loss_fn = find_cls(f"loss_fn.{config['loss_fn'].lower()}")

    # TODO:: add flexibility for best metric (i.e. max, min)
    metric = find_cls(f"metric.{config['metric'].lower()}")


    # Workspace preparation
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(config["output_dir"], model_name, dt_string)
    workspace = Workspace(output_dir, model, loss_fn, metric, optimizer)


    # Train model
    total_epoch = config["epoch"]

    def evaluate(prefix, model, data_loader, loss_fn, metric):
        model.eval()

        total_loss = 0
        total_metric = 0

        model.train()
        for _, (data, target) in enumerate(tqdm(data_loader, desc=f"Evaluating {prefix} dataset")):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = loss_fn(output, target)

            total_loss += loss.item()
            total_metric += metric(output, target)

        log = {
            "loss": total_loss / len(data_loader),
            "metric": total_metric / len(data_loader)
        }

        return log


    train_log = {
        "train_loss": [],
        "train_metric": [],
        "dev_loss": [],
        "dev_metric": [],
        "best_epoch": 0,
        "best_dev_metric": 0,
        "best_dev_loss": 0
    }


    for epoch in tqdm(range(total_epoch), desc="Training"):
        total_loss = 0
        total_metric = 0

        model.train()
        for _, (data, target) in enumerate(train_data_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_metric += metric(output, target)

        train_log["train_loss"].append(total_loss / len(train_data_loader))
        train_log["train_metric"].append(total_metric / len(train_data_loader))

        dev_log = evaluate("dev", model, dev_data_loader, loss_fn, metric)

        train_log["dev_loss"].append(dev_log["loss"])
        train_log["dev_metric"].append(dev_log["metric"])

        if train_log["dev_metric"][-1] > train_log["best_dev_metric"]:
            print("\tsaving the model with the best dev metric")
            train_log["best_epoch"] = epoch
            train_log["best_dev_loss"] = train_log["dev_loss"][-1]
            train_log["best_dev_metric"] = train_log["dev_metric"][-1]

            workspace.save_best_model(train_log)

        if epoch % config["checkpoint_frequency"] == 0:
            workspace.save_checkpoint(epoch, train_log)

            print("epochs {}".format(epoch))
            print("\ttrain_loss: {}".format(train_log["train_loss"][-1]))
            print("\ttrain_metric: {}".format(train_log["train_metric"][-1]))
            print("\tdev_loss: {}".format(train_log["dev_loss"][-1]))
            print("\tdev_metric: {}".format(train_log["dev_metric"][-1]))

    workspace.save_checkpoint(total_epoch-1, train_log)

    # load the best model
    checkpoint = workspace.load_best_model()

    print("Training results")
    print("\tbest_epoch: {}".format(checkpoint["best_epoch"]))
    print("\tbest_dev_loss: {}".format(checkpoint["best_dev_loss"]))
    print("\tbest_dev_metric: {}".format(checkpoint["best_dev_metric"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")

    parser.add_argument("--config", default=None, required=True, type=str,
                        help="path to config file")

    args = parser.parse_args()

    main(load_json(args.config))
