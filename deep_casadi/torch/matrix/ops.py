import torch

from deep_casadi.common.decorator import casadi


@casadi
def vcat(args):
    return torch.cat(args, dim=0)


@casadi
def vertcat(*args):
    return torch.cat(args, dim=0)


@casadi
def hcat(args):
    return torch.cat(args, dim=1)


@casadi
def horzcat(*args):
    return torch.cat(args, dim=1)
