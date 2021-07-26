import torch
import casadi as cs

from deep_casadi.modules import Module


class Sigmoid(Module, torch.nn.Sigmoid):
    def cs_forward(self, x):
        y = 1 / (1 + cs.exp(-x))
        return y


class Tanh(Module, torch.nn.Tanh):
    def cs_forward(self, x):
        return cs.tanh(x)


class ReLU(Module, torch.nn.ReLU):
    def cs_forward(self, x):
        return cs.if_else(x < 0., 0. * x, x)


class LeakyReLUu(Module, torch.nn.LeakyReLU):
    def cs_forward(self, x):
        return cs.if_else(x < 0., self.negative_slope * x, x)
