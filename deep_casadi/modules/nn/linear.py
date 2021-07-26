import torch
import casadi as cs

from deep_casadi.modules import Module


class Linear(Module, torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *args, **kwargs):
        super().__init__(in_features, out_features, bias, *args, **kwargs)

        self.cs_weight = cs.MX.sym("weight", out_features, in_features)
        if bias:
            self.cs_bias = cs.MX.sym("bias", out_features, 1)
        else:
            self.cs_bias = cs.DM(0.)

    def cs_forward(self, x):
        y = cs.mtimes(self.cs_weight, x)
        if self.bias is not None:
            y = y + self.cs_bias
        return y