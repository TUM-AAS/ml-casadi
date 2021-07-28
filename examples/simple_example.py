import numpy as np
import torch
import casadi as cs
import deep_casadi.torch as dc
import matplotlib.pyplot as plt


class Model(dc.TorchDeepCasadiModule):
    def forward(self, x):
        return dc.horzcat(
            dc.sin(x),
            dc.cos(x)
        )


def example():
    model = Model()

    torch_inp = torch.rand((2, 3))
    casadi_inp = cs.DM(torch_inp.numpy())
    print('Model evaluated in Pytorch')
    print(model(torch_inp).detach().numpy())

    print('Model evaluated in Casadi')
    print(model(casadi_inp))

    ## Create a randomly initialized non linear Multi Layer Perceptron
    model2 = dc.nn.MLPerceptron(2, 3, 1, 1, 'Tanh')

    ## Export the model as Casadi Function
    casadi_sym_inp = cs.MX.sym('inp', 2)
    casadi_sym_out = model2(casadi_sym_inp)
    casadi_func = cs.Function('model2',
                              [casadi_sym_inp, model2.params(symbolic=True, approx=False, flat=True)],
                              [casadi_sym_out])
    casadi_param = model2.params(symbolic=False, approx=False, flat=True)

    ## Export a linear approximation of the model as Casadi Function
    model2.input_size = 2
    model2.output_size = 1
    casadi_lin_approx_sym_out = model2.approx(casadi_sym_inp, order=1)
    casadi_lin_approx_func = cs.Function('model2_lin',
                                         [casadi_sym_inp,
                                          model2.params(symbolic=True, approx=True, flat=True, order=1)],
                                         [casadi_lin_approx_sym_out])
    casadi_lin_approx_param = model2.params(symbolic=False, approx=True, flat=True, order=1, a=np.zeros((2,)))

    ## Export a quadratic approximation of the model as Casadi Function
    casadi_quad_approx_sym_out = model2.approx(casadi_sym_inp, order=2)
    casadi_quad_approx_func = cs.Function('model2_quad',
                                          [casadi_sym_inp,
                                           model2.params(symbolic=True, approx=True, flat=True, order=2)],
                                          [casadi_quad_approx_sym_out])
    casadi_quad_approx_param = model2.params(symbolic=False, approx=True, flat=True, order=2, a=np.zeros((2,)))

    ## Evaluate the functions and compare to the torch MLP
    inputs = np.stack([np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)], axis=1)
    torch_out = model2(torch.tensor(inputs).float()).detach().numpy()
    casadi_out = []
    casadi_lin_out = []
    casadi_quad_out = []

    # Casadi can not handle batches
    for i in range(100):
        casadi_out.append(casadi_func(inputs[i], casadi_param))
        casadi_lin_out.append(casadi_lin_approx_func(inputs[i], casadi_lin_approx_param))
        casadi_quad_out.append(casadi_quad_approx_func(inputs[i], casadi_quad_approx_param))

    casadi_out = np.array(casadi_out).squeeze(axis=-1)
    casadi_lin_out = np.array(casadi_lin_out).squeeze(axis=-1)
    casadi_quad_out = np.array(casadi_quad_out).squeeze(axis=-1)

    plt.plot(torch_out, label='Torch', linewidth=4)
    plt.plot(casadi_out, label='Casadi')
    plt.plot(casadi_lin_out, label='Casadi Linear')
    plt.plot(casadi_quad_out, label='Casadi Quadratic')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    example()
