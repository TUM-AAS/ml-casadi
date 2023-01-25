import numpy as np
import torch
import torchvision.models as models
import casadi as cs
import ml_casadi.torch as mc


class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.res_net = models.resnet18()

        self.linear = torch.nn.Linear(1000, 10)

    def forward(self, x):
        return self.linear(self.res_net(x))


class ResNetFlattenInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.res_net = ResNet()

    def forward(self, x):
        x = x.reshape(-1, 3, 64, 64)
        return self.res_net(x)


def example():
    input = torch.rand((3, 64, 64))

    ## Create a randomly initialized non linear Multi Layer Perceptron
    model = mc.TorchMLCasadiModuleWrapper(ResNetFlattenInput(), input_size=3*64*64, output_size=10)

    input_flatten = input.flatten()
    casadi_sym_inp = cs.MX.sym('inp', 3 * 64 * 64)
    casadi_sym_out = model.approx(casadi_sym_inp, order=1)
    casadi_func = cs.Function('model_approx_wrapper',
                              [casadi_sym_inp, model.sym_approx_params(order=1, flat=True)],
                              [casadi_sym_out])
    casadi_param = model.approx_params(input_flatten, flat=True, order=1)

    torch_out = model(torch.tensor(input).float()).detach().numpy()

    casadi_out = np.array(casadi_func(np.array(input_flatten), casadi_param))[:, 0]

    assert np.allclose(torch_out, casadi_out, atol=1e-6)

    print('Model evaluated in Pytorch')
    print(torch_out)

    print('Model evaluated in Casadi')
    print(casadi_out)


if __name__ == '__main__':
    example()
