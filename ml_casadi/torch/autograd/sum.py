import functools
from typing import Callable

import torch
import functorch
import time

def batched_jacobian2(func: Callable, inputs: torch.Tensor, create_graph=False, return_func_output=False):
    r"""Function that computes batches of the Jacobian of a given function and a batch of inputs.

    Args:
        func: a Python function that takes Tensor inputs and returns a tuple of Tensors or a Tensor.
        inputs: inputs to the function ``func``. First dimension is treated as batch dimension
        create_graph: If ``True``, the Jacobian will be computed in a differentiable manner.
        return_func_output: If ``True``, the function output will be returned.

    Returns:Jacobian

    """

    t = time.time()
    def bjvp(func):
        return functools.partial(functorch.jvp, func)
    test = functorch.vmap(bjvp(func))
    t = time.time()
    test((inputs,), (torch.ones_like(inputs),))
    print(f'jvp: {time.time() - t}')

    def aux_function(inputs):
        out = func(inputs)
        return out, out
    with torch.no_grad():
        if not return_func_output:
            return functorch.vmap(functorch.jacrev(func))(inputs)
        return functorch.vmap(functorch.jacrev(aux_function, has_aux=True))(inputs)

def batched_jacobian(func: Callable, inputs: torch.Tensor, create_graph=False, return_func_output=False):
    r"""Function that computes batches of the Jacobian of a given function and a batch of inputs.

    Args:
        func: a Python function that takes Tensor inputs and returns a tuple of Tensors or a Tensor.
        inputs: inputs to the function ``func``. First dimension is treated as batch dimension
        create_graph: If ``True``, the Jacobian will be computed in a differentiable manner.
        return_func_output: If ``True``, the function output will be returned.

    Returns:Jacobian

    """

    inputs.requires_grad = True
    func_output_storage = []

    def func_sum_batch(x: torch.Tensor):
        func_output = func(x)
        func_output_storage.append(func_output if create_graph else func_output.detach())
        return func_output.sum(axis=0)

    jacobian = torch.autograd.functional.jacobian(func_sum_batch,
                                                  inputs,
                                                  vectorize=True,
                                                  create_graph=create_graph
                                                  ).moveaxis(-len(inputs.shape), 0)

    if return_func_output:
        return jacobian, func_output_storage[0]

    return jacobian


def batched_hessian(func: Callable, inputs: torch.Tensor, create_graph=False,
                    return_jacobian=False, return_func_output=False):
    r"""

    Args:
        func: a Python function that takes Tensor inputs and returns a tuple of Tensors or a Tensor.
        inputs: inputs to the function ``func``. First dimension is treated as batch dimension
        create_graph: If ``True``, the Hessian will be computed in a differentiable manner.
        return_jacobian: If ``True``, the Jacobian will be returned.
        return_func_output: If ``True``, the function output will be returned.

    Returns: Hessian

    """
    additional_outputs = []

    def jacobian_func(x: torch.Tensor):
        out = batched_jacobian(func, x, create_graph=True, return_func_output=return_func_output)

        if return_func_output:
            additional_outputs.append(out[1] if create_graph else out[1].detach())
            jacobian = out[0]
        else:
            jacobian = out

        if return_jacobian:
            additional_outputs.insert(0, jacobian if create_graph else jacobian.detach())

        return jacobian

    hessian = batched_jacobian(jacobian_func, inputs)

    if len(additional_outputs) > 0:
        return (hessian if create_graph else hessian.detach(), *additional_outputs)

    return hessian if create_graph else hessian.detach()

# from functorch import vmap, jacrev
# def batched_jacobian(func: Callable, inputs: torch.Tensor, create_graph=False, return_func_output=False):
#     with torch.no_grad():
#         if not return_func_output:
#             return vmap(jacrev(func))(inputs)
#         return vmap(jacrev(func))(inputs), func(inputs)
