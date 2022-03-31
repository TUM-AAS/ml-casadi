import numpy as np
import pytest

import torchvision.models as models
import casadi as cs
import torch
import ml_casadi.torch as mc


class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.res_net = models.resnet18()

        self.linear = torch.nn.Linear(1000, 10)

    def forward(self, x):
        return self.linear(self.res_net(x))


class FlattenedResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.res_net = ResNet()

    def forward(self, x):
        x = x.reshape(-1, 3, 64, 64)
        return self.res_net(x)


class TestModuleWrapper:
    @pytest.fixture
    def model(self):
        return mc.TorchMLCasadiModuleWrapper(FlattenedResNet(), input_size=3*64*64, output_size=10)

    @pytest.fixture
    def input(self):
        return np.random.rand(3, 64, 64)

    def test_module_wrapper(self, model, input):
        input_flatten = input.flatten()
        casadi_sym_inp = cs.MX.sym('inp', 3 * 64 * 64)
        casadi_sym_out = model.approx(casadi_sym_inp, order=1)
        casadi_func = cs.Function('model_approx_wrapper',
                                  [casadi_sym_inp, model.sym_approx_params(order=1, flat=True)],
                                  [casadi_sym_out])
        casadi_param = model.approx_params(input_flatten, flat=True, order=1)

        torch_out = model(torch.tensor(input).float()).detach().numpy()

        casadi_out = np.array(casadi_func(input_flatten, casadi_param))[:, 0]

        assert np.allclose(torch_out, casadi_out, atol=1e-6)