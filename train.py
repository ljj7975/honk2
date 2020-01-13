import argparse
import torch
import numpy as np
import random
import model as model_modules
import data_loader as data_loader_modules
import metric as metric_modules
import loss_function as loss_fn_modules
import torch.optim as optimizer_modules
from tqdm import tqdm
from pprint import pprint
from utils import prepare_device, load_json


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


    # Optimizer
    optimizer_config = config["optimizer"]
    optimizer_config["config"]["params"] = model.parameters()
    optimizer_class = getattr(optimizer_modules, optimizer_config["name"])
    optimizer = optimizer_class(**optimizer_config["config"])

    loss_fn = getattr(loss_fn_modules, config["loss_fn"])
    metric = getattr(metric_modules, config["metric"])


    # Train model
    total_epoch = config["epoch"]

    model.train()
    for epoch in range(total_epoch):
        total_loss = 0
        total_metrics = 0

        for batch_idx, (data, target) in enumerate(training_data_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_metrics += metric(output, target)

        log = {
            'loss': total_loss / len(training_data_loader),
            'metrics': (total_metrics / len(training_data_loader)).tolist()
        }

        pprint(log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Feature Extractor')

    parser.add_argument('--config', default=None, required=True, type=str,
                        help='path to config file')

    args = parser.parse_args()

    main(load_json(args.config))
