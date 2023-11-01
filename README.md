[![ML-CasADi CI](https://github.com/TUM-AAS/ml-casadi/actions/workflows/python-test.yml/badge.svg)](https://github.com/TUM-AAS/ml-casadi/actions/workflows/python-test.yml)

---
# New Framework: L4CasADi
The functionality of ML-CasADi has been merged with [L4CasADi](https://github.com/Tim-Salzmann/l4casadi).

- Approximated: [RealTimeL4CasADi](https://github.com/Tim-Salzmann/l4casadi#real-time-l4casadi).
- Naive: [Naive L4CasADi](https://github.com/Tim-Salzmann/l4casadi#naive-l4casadi)

Additionally L4CasADi enables the use of PyTorch models and functions in a CasADi graph while supporting CasADi code generation 
capabilities. You can find more information [here](https://github.com/Tim-Salzmann/l4casadi).

---
# **Deprecated** ~~ML-CasADi~~
This is the underlying framework enabling Real-time Neural-MPC in our paper
```
Real-time Neural-MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms
```
[Arxiv Link](https://arxiv.org/pdf/2203.07747)

If you are looking for the experimental code you can find it [here](https://github.com/TUM-AAS/neural-mpc).

This framework enables trained PyTorch Models to be used in CasADi graphs and subsequently in Acados optimal control problems.

There are two different ways this framework enables PyTorch models in a CasADi graph:

**Naively**, where the operations of the PyTorch model are reconstructed in the CasADi graph and the learned weights are copied over. This is limited to dense multi-layer perceptrons and can be slow for large networks as CasADi is not optimized for large matrix multiplications.

**Approximated**, where the PyTorch model is abstracted as first or second order approximation. The necessary parameters are passed to the CasADi function at every function call. This enables the use of any differentiable PyTorch module. Our paper describes how the approximation can be used to efficiently apply a learned dynamics model efficiently in an MPC setting.

## Integration with Acados
To use this framework with Acados:
- Follow the [installation instructions](https://docs.acados.org/installation/index.html).
- Install the [Python Interface](https://docs.acados.org/python_interface/index.html).
- Ensure that `LD_LIBRARY_PATH` is set correctly (`DYLD_LIBRARY_PATH`on MacOS).
- Ensure that `ACADOS_SOURCE_DIR` is set correctly.

An example of how a PyTorch model can be used as dynamics model in the Acados framework for Model Predictive Control can be found in `examples/mpc_mlp_cnn_example.py` (Approximated) and `examples/mpc_mlp_naive_example.py` (Naive).

## Functorch
ML-CasADi now supports functorch for batched Jacobian and Hessian calculation if the functorch package. While the batched functorch approach is faster, functorch is not compatible with some PyTorch operations such as `BatchNorm` ([Link](https://pytorch.org/functorch/stable/batch_norm.html)). To use functorch batch differentiation adjust the import in `torch/autograd/__init__.py`.

## Examples
### Approximated
```
import ml_casadi.torch as mc
import casadi as ca
import numpy as np
import torch

size_in = 6
size_out = 3
model = mc.TorchMLCasadiModuleWrapper(
    torch_module,
    input_size=size_in,
    output_size=size_out)
    
casadi_sym_inp = ca.MX.sym('inp',size_in)
casadi_sym_out = model.approx(casadi_sym_inp, order=1)
casadi_func = ca.Function('model_approx_wrapper',
                          [casadi_sym_inp, model.sym_approx_params(order=1, flat=True)],
                          [casadi_sym_out])

inp = np.ones([1, size_in])  # torch needs batch dimension
casadi_param = model.approx_params(inp, order=1, flat=True)  # order=2
casadi_out = casadi_func(inp.transpose(-2, -1), casadi_param)   # transpose for vector rep. expected by casadi

t_out = model(torch.tensor(inp, dtype=torch.float32))

print(casadi_out)
print(t_out)
```

### Naive
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

## Citing
If you use our work please cite our paper
```
@article{salzmann2023neural,
  title={Real-time Neural-MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms},
  author={Salzmann, Tim and Kaufmann, Elia and Arrizabalaga, Jon and Pavone, Marco and Scaramuzza, Davide and Ryll, Markus},
  journal={IEEE Robotics and Automation Letters},
  doi={10.1109/LRA.2023.3246839},
  year={2023}
}
```