import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv2d(1, config["n_feature_maps"], (3, 3), padding=(1, 1), bias=False)
        if "res_pool" in config:
            self.pool = nn.AvgPool2d(config["res_pool"])

        self.n_layers = config["n_layers"]
        self.convs = [nn.Conv2d(config["n_feature_maps"], config["n_feature_maps"], (3, 3), padding=1, dilation=1,
            bias=False) for _ in range(self.n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(config["n_feature_maps"], affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        self.output = nn.Linear(config["n_feature_maps"], config["n_labels"])

    def forward(self, x):
        x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)
