import torch

from deep_casadi.common.decorator import casadi


@casadi
def sin(x):
    return torch.sin(x)


@casadi
def cos(x):
    return torch.cos(x)


@casadi
def tan(x):
    return torch.tan(x)
