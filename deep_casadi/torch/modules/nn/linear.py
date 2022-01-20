import torch
import casadi as cs

from deep_casadi.torch.modules import TorchDeepCasadiModule


class Linear(TorchDeepCasadiModule, torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *args, **kwargs):
        super().__init__(in_features, out_features, bias, *args, **kwargs)

    def cs_forward(self, x):
        assert x.shape[1] == 1, 'Casadi can not handle batches.'
        y = cs.mtimes(self.weight.detach().numpy(), x)
        if self.bias is not None:
            y = y + self.bias.detach().numpy()
        return y
