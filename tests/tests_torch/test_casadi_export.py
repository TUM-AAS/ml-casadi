import numpy as np
import pytest

import casadi as cs
import torch
import ml_casadi.torch as mc


class Model(mc.TorchMLCasadiModule):
    def __init__(self):
        super().__init__()
        self.register_parameter('m', torch.nn.Parameter(torch.rand((1,))))
        self.register_parameter('b', torch.nn.Parameter(torch.rand((1,))))

    def forward(self, x):
        return self.m * (x - self.b) ** 2

    def cs_forward(self, x):
        return self.m.detach().numpy() * (x - self.b.detach().numpy()) ** 2


class TestCasadiExport:
    @pytest.fixture
    def model(self):
        return Model()

    @pytest.fixture
    def input(self):
        return np.linspace(-5, 5, 101)

    def test_casadi_function(self, model, input):
        casadi_sym_inp = cs.MX.sym('inp', 1)
        casadi_sym_out = model(casadi_sym_inp)
        casadi_func = cs.Function('model',
                                  [casadi_sym_inp],
                                  [casadi_sym_out])

        torch_out = model(torch.tensor(input).float()).detach().numpy()

        casadi_out = []
        for i in range(input.shape[0]):
            casadi_out.append(casadi_func(input[i]))

        casadi_out = np.array(casadi_out).squeeze()

        assert np.allclose(torch_out, casadi_out, atol=1e-6)

    def test_approx_first_order_casadi_function(self, model, input):
        model.input_size = 1
        model.output_size = 1
        order = 1
        casadi_sym_inp = cs.MX.sym('inp', 1)
        casadi_sym_out = model.approx(casadi_sym_inp, order=order)
        casadi_func = cs.Function('model_approx',
                                  [casadi_sym_inp, model.sym_approx_params(order=order, flat=True)],
                                  [casadi_sym_out])
        casadi_param = model.approx_params(np.array([0.]), flat=True, order=order)

        torch_out = model(torch.tensor(input).float()).detach().numpy()

        casadi_out = []
        for i in range(input.shape[0]):
            casadi_out.append(casadi_func(input[i], casadi_param))

        casadi_out = np.array(casadi_out).squeeze()

        assert not np.allclose(torch_out, casadi_out, atol=1e-6)

    def test_approx_second_order_casadi_function(self, model, input):
        model.input_size = 1
        model.output_size = 1
        order = 2
        casadi_sym_inp = cs.MX.sym('inp', 1)
        casadi_sym_out = model.approx(casadi_sym_inp, order=order)
        casadi_func = cs.Function('model_approx',
                                  [casadi_sym_inp, model.sym_approx_params(order=order, flat=True)],
                                  [casadi_sym_out])
        casadi_param = model.approx_params(np.array([0.]), flat=True, order=order)

        torch_out = model(torch.tensor(input).float()).detach().numpy()

        casadi_out = []
        for i in range(input.shape[0]):
            casadi_out.append(casadi_func(input[i], casadi_param))

        casadi_out = np.array(casadi_out).squeeze()

        assert np.allclose(torch_out, casadi_out, atol=1e-6)

    def test_approx_second_order_parallel_casadi_function(self, model, input):
        model.input_size = 1
        model.output_size = 1
        order = 2
        casadi_sym_inp = cs.MX.sym('inp', 1)
        casadi_sym_out = model.approx(casadi_sym_inp, order=order, parallel=True)
        casadi_func = cs.Function('model_approx',
                                  [casadi_sym_inp, model.sym_approx_params(order=order, flat=True)],
                                  [casadi_sym_out])
        casadi_param = model.approx_params(np.array([0.]), flat=True, order=order)

        torch_out = model(torch.tensor(input).float()).detach().numpy()

        casadi_out = []
        for i in range(input.shape[0]):
            casadi_out.append(casadi_func(input[i], casadi_param))

        casadi_out = np.array(casadi_out).squeeze()

        assert np.allclose(torch_out, casadi_out, atol=1e-6)
