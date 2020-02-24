import torch.nn as nn
from utils import register_cls


@register_cls('loss_fn.ce_loss')
def ce_loss(output, target):
    # categorical cross entropy = softmax + nll
    return nn.CrossEntropyLoss()(output, target)

@register_cls('loss_fn.nll_loss')
def nll_loss(output, target):
    return nn.NLLLoss()(output, target)

@register_cls('loss_fn.bce_loss')
def bce_loss(output, target):
    return nn.BCELoss()(output, target)

@register_cls('loss_fn.logsoftmax_nll_loss')
def logsoftmax_nll_loss(output, target):
    return nn.NLLLoss()(nn.LogSoftmax()(output), target.max(1)[1].long())

@register_cls('loss_fn.softmax_bce_loss')
def softmax_bce_loss(output, target):
    return nn.BCELoss()(nn.Softmax()(output), target)

@register_cls('loss_fn.sigmoid_bce_loss')
def sigmoid_bce_loss(output, target):
    return nn.BCELoss()(nn.Sigmoid()(output), target)
