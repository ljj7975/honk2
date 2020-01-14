import argparse
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
from utils import prepare_device, load_json, ensure_dir


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
    model_config = config["model"]
    model_class = getattr(model_modules, model_config["name"])
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
    training_data_loader_config = config["training_data_loader"]
    training_data_loader_class = getattr(data_loader_modules, training_data_loader_config["name"])
    training_data_loader = training_data_loader_class(training_data_loader_config["config"])
    print(f"train data loader: {training_data_loader}")

    test_data_loader_config = config["test_data_loader"]
    test_data_loader_class = getattr(data_loader_modules, test_data_loader_config["name"])
    test_data_loader = test_data_loader_class(test_data_loader_config["config"])
    print(f"test data loader: {test_data_loader}")


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

        training_log = {
            "loss" : [],
            "metric" : []
        }

        for epoch in tqdm(range(total_epoch)):
            total_loss = 0
            total_metric = 0

            for _, (data, target) in enumerate(training_data_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_metric += metric(output, target)

            training_log["loss"].append(total_loss / len(training_data_loader))
            training_log["metric"].append(total_metric / len(training_data_loader))

            if epoch % config["checkpoint_frequency"] == 0:
                torch.save(model.state_dict(), checkpoint_path_template.format(epoch))

                print(f"epochs {epoch}")
                print(f"\tloss: {training_log['loss'][-1]}")
                print(f"\tmetric: {training_log['metric'][-1]}")

                if training_log["metric"][-1] > best_metric:
                    print("\tsaving the model with the best metric")
                    torch.save(model.state_dict(), best_model_path)
                    best_epoch = epoch
                    best_loss = training_log["loss"][-1]
                    best_metric = training_log["metric"][-1]

        print("Training results")
        print(f"\tbest_epoch: {best_epoch}")
        print(f"\tbest_loss: {best_loss}")
        print(f"\tbest_metric: {best_metric}")


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Feature Extractor')

    parser.add_argument('--config', default=None, required=True, type=str,
                        help='path to config file')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'], help='whether to train a new model or not (default: train)')

    args = parser.parse_args()

    main(args.mode, load_json(args.config))
