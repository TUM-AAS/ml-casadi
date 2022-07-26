![CI](https://github.com/Tim-Salzmann/deep_casadi/actions/workflows/python-test.yml/badge.svg)

# ML-CasADi
This is the underlying framework enabling Neural-MPC in our paper

`Neural-MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms`

[Arxiv Link](https://arxiv.org/pdf/2203.07747)

This framework enables PyTorch Models to be used as CasADi functions and subsequently in Acados optimal control problems.

## Examples
### Arbitrary PyTorch Model as first- or second order approximation
```
import ml_casadi.torch as mc
import casadi as cs

model = mc.TorchMLCasadiModuleWrapper(
    torch_module,
    input_size=size_in,
    output_size=size_out)
    
casadi_sym_inp = cs.MX.sym('inp', size_in)
casadi_sym_out = model.approx(casadi_sym_inp, order=1)  # order=2

casadi_func = cs.Function('model_approx_wrapper',
                          [casadi_sym_inp, model.sym_approx_params(order=1)],
                          [casadi_sym_out])

casadi_param = model.approx_params(input, order=1)  # order=2
casadi_out = casadi_func(input, casadi_param)
```

Using Acados, the approximation parameters can be passed to the optimal control problem via `acados_ocp_solver.set(n, 'p', casadi_param)`.

### Specific PyTorch Model without approximation
```
import ml_casadi.torch as mc
import casadi as cs

model = mc.MultiLayerPerceptron(
    input_size=size_in,
    hidden_size=hidden_size,
    output_size=size_out,
    n_hidden=n_hidden,
    activation='relu')
    
casadi_sym_inp = cs.MX.sym('inp', size_in)
casadi_sym_out = model(casadi_sym_inp)

casadi_func = cs.Function('model_approx_wrapper',
                          [casadi_sym_inp],
                          [casadi_sym_out])

casadi_out = casadi_func(input, casadi_param)
```