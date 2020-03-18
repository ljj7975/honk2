import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import register_cls


@register_cls('model.ResNet')
class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config["n_layers"]
        n_maps = config["n_feature_maps"]

        self.layers = nn.ModuleDict()

        self.layers["conv_0"] = nn.Conv2d(1, n_maps, (3, 3), padding=1, bias=False)

        for i in range(1, self.n_layers + 1):
            if config["use_dilation"]:
                padding_size = int(2**((i-1) // 3))
                dilation_size = int(2**((i-1) // 3))
                self.layers[f"conv_{i}"] = nn.Conv2d(n_maps, n_maps, (3, 3), padding=padding_size, dilation=dilation_size, bias=False)
            else:
                self.layers[f"conv_{i}"] = nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, bias=False)
            self.layers[f"bn_{i}"] = nn.BatchNorm2d(n_maps, affine=False)

        if "avg_pool" in config:
            self.layers["avg_pool"] = nn.AvgPool2d(config["avg_pool"])

        self.layers["output"] = nn.Linear(n_maps, config["n_labels"])

        self.activations = nn.ModuleDict({
            "relu": nn.ReLU()
        })

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layers["conv_0"](x)
        x = self.activations["relu"](x)

        if "avg_pool" in self.layers:
            x = self.layers["avg_pool"](x)

        prev_x = x
        for i in range(1, self.n_layers + 1):
            x = self.layers[f"conv_{i}"](x)
            x = self.activations["relu"](x)

            if i % 2 == 0:
                x = x + prev_x
                prev_x = x

            x = self.layers[f"bn_{i}"](x)

        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, features, o3)
        x = x.mean(2)
        x = self.layers["output"](x)
        return x
