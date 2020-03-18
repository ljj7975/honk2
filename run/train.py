import argparse
import os
import random
from datetime import datetime

import torch
import torch.optim as optimizer_modules
import torch.optim.lr_scheduler as lr_scheduler_modules
from tqdm import tqdm

from .runnable_utils import merge_configs, init_data_loader, set_seed
from .test import evaluate
from utils import DatasetType, Workspace
from utils import find_cls, load_json, prepare_device


def log_process(prefix, writer, epoch, log):
    for key, val in log.items():
        writer.add_scalar(f'{prefix}/{key}', val, epoch)


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


    # Prepare DataLoader
    train_data_loader = init_data_loader(config, DatasetType.TRAIN)
    print(f"training dataset size: {len(train_data_loader.dataset)}")

    dev_data_loader = init_data_loader(config, DatasetType.DEV)
    print(f"dev dataset size: {len(dev_data_loader.dataset)}")

    test_data_loader = init_data_loader(config, DatasetType.TEST)
    print(f"test dataset size: {len(test_data_loader.dataset)}")


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

    loss_fn = find_cls(f"loss_fn.{config['loss_fn'].lower()}")

    # TODO:: add flexibility for best metric (i.e. max, min)
    metric = find_cls(f"metric.{config['metric'].lower()}")

    # Workspace preparation
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(config["output_dir"], model_name, dt_string)
    workspace = Workspace(device, output_dir, model, loss_fn, metric, optimizer, lr_scheduler)

    # store meta data
    writer = workspace.summary_writer
    writer.add_scalar('Meta/Parameters', sum(p.numel() for p in params))


    # Train model
    total_epoch = config["epoch"]

    log = {
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
            if use_per_epoch_stepping:
                lr_scheduler.step()

            total_loss += loss.item()
            total_metric += metric(output, target)

        train_results = {
            "loss": total_loss / len(train_data_loader),
            "metric": total_metric / len(train_data_loader)
        }

        log_process('Train', writer, epoch, train_results)

        dev_results = evaluate(device, "Dev", model, dev_data_loader, loss_fn, metric)

        log_process('Dev', writer, epoch, train_results)

        print("epochs {}".format(epoch))
        print("learning rate: {}".format(lr_scheduler.get_lr()))
        print("\ttrain_loss: {}".format(train_results["loss"]))
        print("\ttrain_metric: {}".format(train_results["metric"]))
        print("\tdev_loss: {}".format(dev_results["loss"]))
        print("\tdev_metric: {}".format(dev_results["metric"]))

        # TODO :: different inequality must be used depending on the type of metric
        if dev_results["metric"] > log["best_dev_metric"]:
            print("\tsaving the model of the best dev metric")
            log["best_epoch"] = epoch
            log["best_dev_loss"] = dev_results["loss"]
            log["best_dev_metric"] = dev_results["metric"]

            workspace.save_best_model(log)

        # TODO:: create best model at the first iteration to make sure it exists

        if epoch % config["checkpoint_frequency"] == 0:
            workspace.save_checkpoint(epoch, log)

        if not use_per_epoch_stepping:
            lr_scheduler.step()

    workspace.save_checkpoint(total_epoch-1, log)

    # load the best model
    checkpoint = workspace.load_best_model()

    print("Training results")
    print("\tbest_epoch: {}".format(checkpoint["best_epoch"]))
    print("\tbest_dev_loss: {}".format(checkpoint["best_dev_loss"]))
    print("\tbest_dev_metric: {}".format(checkpoint["best_dev_metric"]))


    # Test model
    test_log = evaluate(device, "Test", model, test_data_loader, loss_fn, metric)

    print("Test results")
    print("\ttest_loss: {}".format(test_log["loss"]))
    print("\ttest_metric: {}".format(test_log["metric"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")

    parser.add_argument("--config", default=None, required=True, type=str,
                        help="path to config file")

    args = parser.parse_args()

    main(load_json(args.config))
