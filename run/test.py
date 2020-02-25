import argparse
import copy
import torch
import numpy as np
import os
import random
import model as model_modules
import data_loader as data_loader_modules
import metric as metric_modules
import loss_function as loss_fn_modules
import torch.optim as optimizer_modules
from datetime import datetime
from tqdm import tqdm
from utils import prepare_device, load_json, DatasetType, Workspace


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

    data_loader_class = getattr(data_loader_modules, data_loader_name)
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
    model_class = getattr(model_modules, model_config["name"])

    model_config["config"]["n_labels"] = n_labels
    model = model_class(model_config["config"])
    model_name = type(model).__name__


    # Multiple GPU supports
    if len(gpu_device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_device_ids)

    model.to(device)
    print("model:\n", model)


    # Prepare DataLoader
    test_data_loader = init_data_loader(config, DatasetType.TEST)
    print(f"test dataset size: {len(test_data_loader.dataset)}")


    loss_fn = getattr(loss_fn_modules, config["loss_fn"])

    # TODO:: support multiple metric
    # TODO:: add flexibility for best metric (i.e. max, min)
    metric = getattr(metric_modules, config["metric"])


    # Workspace preparation
    trained_model_dir = config["trained_model_dir"]
    workspace = Workspace(trained_model_dir, model, loss_fn, metric)

    # load the best model
    checkpoint = workspace.load_best_model()

    print("Training results")
    print("\tbest_epoch: {}".format(checkpoint["best_epoch"]))
    print("\tbest_loss: {}".format(checkpoint["best_loss"]))
    print("\tbest_metric: {}".format(checkpoint["best_metric"]))


    # TODO:: support mode == 'test' case (or create explicit runnable script for test mode)

    model.eval()

    total_loss = 0
    total_metric = 0

    for batch_idx, (data, target) in enumerate(tqdm(test_data_loader)):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = loss_fn(output, target)

        total_loss += loss.item()
        total_metric += metric(output, target)


    print("Test results")
    print(f"\tloss: {total_loss / len(test_data_loader)}")
    print(f"\tmetric: {total_metric / len(test_data_loader)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")

    parser.add_argument("--config", default=None, required=True, type=str,
                        help="path to config file")

    args = parser.parse_args()

    main(load_json(args.config))
