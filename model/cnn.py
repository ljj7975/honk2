import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import BaseModel
from utils import register_cls


@register_cls('model.CNN')
class CNN(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleDict()

        conv_in_channels = 1
        conv_out_channels = config["conv_0"]["out_channels"]
        conv_kernel_size = config["conv_0"]["kernel_size"]
        conv_stride_size = config["conv_0"]["stride_size"]

        self.layers["conv_0"] = nn.Conv2d(conv_in_channels, conv_out_channels, conv_kernel_size, stride=conv_stride_size)

        pool_size = config["conv_0"]["pool_size"]
        self.layers["pool_0"] = nn.MaxPool2d(pool_size)

        x = torch.zeros(1, 1, config['height'], config['width'])
        x = self.layers["conv_0"](x)
        x = self.layers["pool_0"](x)
        net_out_size = x.view(1, -1).size(1)

        if "conv_1" in config:
            conv_in_channels = conv_out_channels
            conv_out_channels = config["conv_1"]["out_channels"]
            conv_kernel_size = config["conv_1"]["kernel_size"]
            conv_stride_size = config["conv_1"]["stride_size"]

            self.layers["conv_1"] = nn.Conv2d(conv_in_channels, conv_out_channels, conv_kernel_size, stride=conv_stride_size)

            pool_size = config["conv_1"]["pool_size"]
            self.layers["pool_1"] = nn.MaxPool2d(pool_size)

            x = self.layers["conv_1"](x)
            x = self.layers["pool_1"](x)
            net_out_size = x.view(1, -1).size(1)

        dnn_in_features = 32
        self.layers["lin_0"] = nn.Linear(net_out_size, dnn_in_features)

        if "dnn_0" in config:
            dnn_out_features = config["dnn_0"]["out_features"]

            self.layers["dnn_0"] = nn.Linear(dnn_in_features, dnn_out_features)

            net_out_size = dnn_out_features

            if "dnn_1" in config:
                dnn_in_features = dnn_out_features
                dnn_out_features = config["dnn_1"]["out_features"]

                self.layers["dnn_1"]= nn.Linear(dnn_in_features, dnn_out_features)

                net_out_size = dnn_out_features

        self.layers["lin_1"] = nn.Linear(net_out_size, config["n_labels"])

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
