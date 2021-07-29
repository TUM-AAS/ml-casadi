import torch
import casadi as cs

from deep_casadi.torch.modules import TorchDeepCasadiModule


class Sigmoid(TorchDeepCasadiModule, torch.nn.Sigmoid):
    def cs_forward(self, x):
        y = 1 / (1 + cs.exp(-x))
        return y


class Tanh(TorchDeepCasadiModule, torch.nn.Tanh):
    def cs_forward(self, x):
        return cs.tanh(x)


class ReLU(TorchDeepCasadiModule, torch.nn.ReLU):
    def cs_forward(self, x):
        return cs.if_else(x < 0., 0. * x, x)


class LeakyReLU(TorchDeepCasadiModule, torch.nn.LeakyReLU):
    def cs_forward(self, x):
        return cs.if_else(x < 0., self.negative_slope * x, x)
