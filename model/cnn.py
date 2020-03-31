import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model_utils import BaseModel
from utils import register_cls, calculate_conv_output_size, calculate_pool_output_size


@register_cls('model.CNN')
class CNN(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleDict()

        time = config['time']
        frequency = config['frequency']

        conv_in_channels = 1
        conv_out_channels = config["conv_0"]["out_channels"]
        conv_kernel_size = config["conv_0"]["kernel_size"]
        conv_stride = config["conv_0"]["stride"]

        tensor_size = [conv_in_channels, time, frequency]

        self.layers["conv_0"] = nn.Conv2d(conv_in_channels, conv_out_channels, conv_kernel_size, stride=conv_stride)
        tensor_size = [conv_out_channels] + calculate_conv_output_size(tensor_size[1:], conv_kernel_size, stride=conv_stride)

        pool_kernel_size = config["pool_0"]["kernel_size"]
        self.layers["pool_0"] = nn.MaxPool2d(pool_kernel_size)
        tensor_size = [conv_out_channels] + calculate_pool_output_size(tensor_size[1:], pool_kernel_size)

        if "conv_1" in config:
            conv_in_channels = conv_out_channels
            conv_out_channels = config["conv_1"]["out_channels"]
            conv_kernel_size = config["conv_1"]["kernel_size"]
            conv_stride = config["conv_1"]["stride"]

            self.layers["conv_1"] = nn.Conv2d(conv_in_channels, conv_out_channels, conv_kernel_size, stride=conv_stride)
            tensor_size = [conv_out_channels] + calculate_conv_output_size(tensor_size[1:], conv_kernel_size, stride=conv_stride)

            pool_kernel_size = config["pool_1"]["kernel_size"]
            self.layers["pool_1"] = nn.MaxPool2d(pool_kernel_size)
            tensor_size = [conv_out_channels] + calculate_pool_output_size(tensor_size[1:], pool_kernel_size)

        dnn_in_features = np.prod(tensor_size)
        dnn_out_features = config["lin_0"]["out_features"]

        self.layers["lin_0"] = nn.Linear(dnn_in_features, dnn_out_features)
        
        dnn_in_features = dnn_out_features
        dnn_out_features = config["dnn_0"]["out_features"]

        self.layers["dnn_0"] = nn.Linear(dnn_in_features, dnn_out_features)

        if "dnn_1" in config:
            dnn_in_features = dnn_out_features
            dnn_out_features = config["dnn_1"]["out_features"]

            self.layers["dnn_1"]= nn.Linear(dnn_in_features, dnn_out_features)

        self.layers["lin_1"] = nn.Linear(dnn_out_features, config["n_labels"])

        self.layers["dropout"] = nn.Dropout(config["dropout_prob"])

        self.activations = nn.ModuleDict({
            "relu": nn.ReLU()
        })

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.layers["conv_0"](x)
        x = self.activations["relu"](x)
        x = self.layers["dropout"](x)
        x = self.layers["pool_0"](x)

        if "conv_1" in self.layers:
            x = self.layers["conv_1"](x)
            x = self.activations["relu"](x)
            x = self.layers["dropout"](x)
            x = self.layers["pool_1"](x)

        x = x.view(x.size(0), -1) # shape: (batch, net_out_size)

        x = self.layers["lin_0"](x)

        if "dnn_0" in self.layers:
            x = self.layers["dnn_0"](x)
            x = self.activations["relu"](x)
            x = self.layers["dropout"](x)

        if "dnn_1" in self.layers:
            x = self.layers["dnn_1"](x)
            x = self.layers["dropout"](x)

        x = self.layers["lin_1"](x)
        return x
