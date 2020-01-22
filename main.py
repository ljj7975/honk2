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
from utils import prepare_device, load_json, ensure_dir, DatasetType


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

def main(mode, config):

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


    # Prepare directory for trained model
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
    config["output_dir"] = os.path.join(config["output_dir"], type(model).__name__, dt_string)
    ensure_dir(config["output_dir"])

    checkpoint_path_template = config["output_dir"] + "/checkpoint_{}.pt"
    best_model_path = config["output_dir"] + "/best_model.pt"


    # Optimizer
    optimizer_config = config["optimizer"]
    optimizer_config["config"]["params"] = model.parameters()
    optimizer_class = getattr(optimizer_modules, optimizer_config["name"])
    optimizer = optimizer_class(**optimizer_config["config"])

    loss_fn = getattr(loss_fn_modules, config["loss_fn"])

    # TODO:: support multiple metric
    metric = getattr(metric_modules, config["metric"])


    if mode == "train":

        # Train model
        total_epoch = config["epoch"]

        model.train()

        # TODO:: add flexibility for best metric (i.e. max, min)
        best_epoch = 0
        best_metric = 0
        best_loss = 0

        train_log = {
            "loss" : [],
            "metric" : []
        }

        for epoch in tqdm(range(total_epoch)):
            total_loss = 0
            total_metric = 0

            for _, (data, target) in enumerate(train_data_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_metric += metric(output, target)

            train_log["loss"].append(total_loss / len(train_data_loader))
            train_log["metric"].append(total_metric / len(train_data_loader))

            if epoch % config["checkpoint_frequency"] == 0:
                torch.save(model.state_dict(), checkpoint_path_template.format(epoch))

                print("epochs {0}".format(epoch))
                print("\tloss: {0}".format(train_log["loss"][-1]))
                print("\tmetric: {0}".format(train_log["metric"][-1]))

                if train_log["metric"][-1] > best_metric:
                    print("\tsaving the model with the best metric")
                    torch.save(model.state_dict(), best_model_path)
                    best_epoch = epoch
                    best_loss = train_log["loss"][-1]
                    best_metric = train_log["metric"][-1]

        print("Training results")
        print("\tbest_epoch: {0}".format(best_epoch))
        print("\tbest_loss: {0}".format(best_loss))
        print("\tbest_metric: {0}".format(best_metric))


    # For train mode, evalaute the best model
    if mode == "train" and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

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
    print(f"\tbest_metric: {total_metric / len(test_data_loader)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")

    parser.add_argument("--config", default=None, required=True, type=str,
                        help="path to config file")

    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test"], help="whether to train a new model or not (default: train)")

    args = parser.parse_args()

    main(args.mode, load_json(args.config))
