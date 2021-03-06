import argparse
import operator
import os
import random
from datetime import datetime
from pprint import pprint

import torch
import torch.optim as optimizer_modules
import torch.optim.lr_scheduler as lr_scheduler_modules
from tqdm import tqdm

from .run_utils import merge_configs, init_data_loader, set_seed
from .test import evaluate
from dataset import DatasetType
from metric import MetricType, collect_metrics
from utils import Workspace, find_cls, load_json, prepare_device, num_floats_to_GB


def log_process(prefix, writer, epoch, log):
    for key, val in log.items():
        if type(val) is dict:
            writer.add_scalars(f'{prefix}/{key}', val, epoch)
        else:
            writer.add_scalar(f'{prefix}/{key}', val, epoch)


def main(config):
    set_seed(config["seed"])

    config_name = config["name"]
    print(f"config name: {config_name}")

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

    num_params = model.num_params()
    print("number of parameters: "
        + f"{num_params} "
        + f"({num_floats_to_GB(num_params)} GB)")
    num_trainable_params = model.num_trainable_params()
    print("number of trainable parameters: "
        + f"{num_trainable_params} "
        + f"({num_floats_to_GB(num_trainable_params)} GB)")


    # Prepare DataLoader
    train_data_loader = init_data_loader(config, DatasetType.TRAIN)
    print(f"training dataset size: {len(train_data_loader.dataset)}")

    dev_data_loader = init_data_loader(config, DatasetType.DEV)
    print(f"dev dataset size: {len(dev_data_loader.dataset)}")

    test_data_loader = init_data_loader(config, DatasetType.TEST)
    print(f"test dataset size: {len(test_data_loader.dataset)}")

    label_mapping = train_data_loader.dataset.label_mapping


    # Model training initialization
    optimizer_config = config["optimizer"]
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer_config["config"]["params"] = params
    optimizer_class = getattr(optimizer_modules, optimizer_config["name"])
    optimizer = optimizer_class(**optimizer_config["config"])

    lr_scheduler_config = config["lr_scheduler"]
    lr_scheduler_class = getattr(lr_scheduler_modules, lr_scheduler_config["name"])
    use_per_epoch_stepping = lr_scheduler_config["config"].pop("use_per_epoch_stepping")
    lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_config["config"])

    loss_fn = find_cls(f"loss_fn.{config['loss_fn']}")

    metrics = {}
    for metric_name in config['metric']:
        metrics[metric_name] = find_cls(f"metric.{metric_name}")()

    criterion = metrics[config['criterion'][0]]
    assert criterion.get_type() == MetricType.MACRO

    if config['criterion'][1] == "max":
        criterion_operator = operator.ge
    elif config['criterion'][1] == "min":
        criterion_operator = operator.le


    # Workspace preparation
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")
    output_dir = os.path.join(config["output_dir"], model_name, f"{config_name}_{dt_string}")
    workspace = Workspace(device, output_dir, model, loss_fn, metrics, optimizer, lr_scheduler)

    # store meta data
    writer = workspace.summary_writer
    writer.add_scalar('Meta/Parameters', num_params)
    writer.add_scalar('Meta/TrainableParameters', num_trainable_params)


    # Train model
    total_epoch = config["epoch"]

    log = {
        "best_epoch": 0,
        "best_dev_criterion": 0,
        "best_dev_loss": 0
    }

    for epoch in tqdm(range(total_epoch), desc="Training"):
        total_loss = 0

        model.train()
        for _, (data, target) in enumerate(tqdm(train_data_loader, desc=f"Epoch {epoch}")):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if use_per_epoch_stepping:
                lr_scheduler.step()

            total_loss += loss.item()

            for metric in metrics.values():
                metric.accumulate(output, target)

        train_results = {
            "loss": total_loss / len(train_data_loader),
            "learning_rate": lr_scheduler.get_lr()[0]
        }

        train_results.update(collect_metrics(metrics, label_mapping))

        log_process('Train', writer, epoch, train_results)

        dev_results = evaluate(device, "Dev", model, dev_data_loader, loss_fn, metrics, label_mapping)

        log_process('Dev', writer, epoch, dev_results)

        print("epochs {}".format(epoch))
        print("learning rate: {}".format(train_results["learning_rate"]))
        print("< train results >")
        pprint(train_results)
        print("< dev results >")
        pprint(dev_results)


        if criterion_operator(criterion.get_metric(), log["best_dev_criterion"]):
            print("\tsaving the model of the best dev metric")
            log["best_epoch"] = epoch
            log["best_dev_criterion"] = criterion.get_metric()
            log["best_dev_loss"] = dev_results["loss"]

            workspace.save_best_model(log)

        if epoch % config["checkpoint_frequency"] == 0:
            workspace.save_checkpoint(epoch, log)

        if not use_per_epoch_stepping:
            lr_scheduler.step()

        for metric in metrics.values():
            metric.reset_metric()

    workspace.save_checkpoint(total_epoch-1, log)

    # load the best model
    checkpoint = workspace.load_best_model()

    print("Training results")
    print("\tbest_epoch: {}".format(checkpoint["best_epoch"]))
    print("\tbest_dev_loss: {}".format(checkpoint["best_dev_loss"]))
    print("\tbest_dev_criterion: {}".format(checkpoint["best_dev_criterion"]))


    # Test model
    test_results = evaluate(device, "Test", model, test_data_loader, loss_fn, metrics, label_mapping)

    print("< test results >")
    pprint(test_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")

    parser.add_argument("--config", default=None, required=True, type=str,
                        help="path to config file")

    args = parser.parse_args()

    main(load_json(args.config))
