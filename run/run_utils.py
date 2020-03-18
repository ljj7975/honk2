import copy
import random

import numpy as np
import torch

import dataset as dataset_modules
import data_loader as data_loader_modules
import model as model_modules
import metric as metric_modules
import loss_function as loss_fn_modules
from utils import find_cls


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def merge_configs(base_config, additional_config):
    new_config = copy.deepcopy(base_config)
    for key, value in additional_config.items():
        new_config[key] = value

    return new_config

def init_data_loader(config, type):
    # Initialize dataset
    type_key = type.value
    dataset_name = config["datasets"][type_key]["dataset"]["name"]
    dataset_config = config["datasets"][type_key]["dataset"]["config"]
    dataset_config = merge_configs(config[dataset_name], dataset_config)
    dataset_config["target_class"] = config["target_class"]
    dataset_config["unknown_class"] = config["unknown_class"]
    dataset_config["silence_class"] = config["silence_class"]
    dataset_config["type"] = type

    dataset_class = find_cls(f"dataset.{dataset_name}")
    dataset = dataset_class(dataset_config)

    # Initialize data_loader
    data_loader_name = config["datasets"][type_key]["data_loader"]["name"]
    data_loader_config = config["datasets"][type_key]["data_loader"]["config"]
    data_loader_config = merge_configs(config[data_loader_name], data_loader_config)

    data_loader_class = find_cls(f"data_loader.{data_loader_name}")
    data_loader = data_loader_class(data_loader_config, dataset)

    return data_loader
