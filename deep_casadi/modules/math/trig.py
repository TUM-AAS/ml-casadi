import torch
import casadi as cs

from deep_casadi.modules import Module


class Sin(Module):
    def forward(self, input):
        if type(input) is cs.MX:
            return self.cs_forward(input)
        return torch.sin(input)

    def cs_forward(self, x):
        return cs.sin(x)


class Cos(Module):
    def forward(self, input):
        if type(input) is cs.MX:
            return self.cs_forward(input)
        return torch.cos(input)

    def cs_forward(self, x):
        return cs.cos(x)


class Tan(Module):
    def forward(self, input):
        if type(input) is cs.MX:
            return self.cs_forward(input)
        return torch.tan(input)

    def cs_forward(self, x):
        return cs.tan(x)