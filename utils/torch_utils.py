import math
import torch
from itertools import repeat
from collections.abc import Iterable

import utils.color_print as cp


def prepare_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        cp.print_color(cp.ColorEnum.YELLOW, "There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        cp.print_color(cp.ColorEnum.YELLOW, "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def generate_tuple(x, n):
    if isinstance(x, Iterable):
        return x
    return tuple(repeat(x, n))

def calculate_conv_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):

    dimension = len(input_size)
    stride = generate_tuple(stride, dimension)
    padding = generate_tuple(padding, dimension)
    dilation = generate_tuple(dilation, dimension)

    output_size = []
    for idx, in_size in enumerate(input_size):
        out_size = in_size + 2 * padding[idx] - (dilation[idx] * (kernel_size[idx] - 1) + 1)
        out_size /= stride[idx]
        out_size = math.floor(out_size + 1)
        output_size.append(out_size)

    return output_size

def calculate_pool_output_size(input_size, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):

    dimension = len(input_size)
    if stride is None:
        stride = kernel_size

    stride = generate_tuple(stride, dimension)
    padding = generate_tuple(padding, dimension)
    dilation = generate_tuple(dilation, dimension)

    output_size = []
    for idx, in_size in enumerate(input_size):
        out_size = in_size + 2 * padding[idx] - (dilation[idx] * (kernel_size[idx] - 1) + 1)
        out_size /= stride[idx]
        if ceil_mode:
            out_size = math.ceil(out_size + 1)
        else:
            out_size = math.floor(out_size + 1)
        output_size.append(out_size)

    return output_size
