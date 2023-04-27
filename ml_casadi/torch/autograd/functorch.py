from typing import Callable

import torch
import functorch


def aux_function(func):
    def inner_aux(inputs):
        out = func(inputs)
        return out, out

    return inner_aux


def batched_jacobian(func: Callable, inputs: torch.Tensor, create_graph=False, return_func_output=False):
    r"""Function that computes batches of the Jacobian of a given function and a batch of inputs.

    Args:
        func: a Python function that takes Tensor inputs and returns a tuple of Tensors or a Tensor.
        inputs: inputs to the function ``func``. First dimension is treated as batch dimension
        create_graph: If ``True``, the Jacobian will be computed in a differentiable manner.
        return_func_output: If ``True``, the function output will be returned.

    Returns:Jacobian

    """

    if not create_graph:
        with torch.no_grad():
            if not return_func_output:
                return functorch.vmap(functorch.jacrev(func))(inputs)
            return functorch.vmap(functorch.jacrev(aux_function(func), has_aux=True))(inputs)
    else:
        if not return_func_output:
            return functorch.vmap(functorch.jacrev(func))(inputs)
        return functorch.vmap(functorch.jacrev(aux_function(func), has_aux=True))(inputs)


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
    def aux_function_jac(func):
        def inner_aux(inputs):
            out = func(inputs)
            return out[0], (out[0], out[1])
        return inner_aux

    if not create_graph:
        with torch.no_grad():
            if not return_func_output and not return_jacobian:
                return functorch.vmap(functorch.jacrev(functorch.jacrev(func)))(inputs)
            elif not return_func_output and return_jacobian:
                return functorch.vmap(functorch.jacrev(aux_function_jac(functorch.jacrev(func)), has_aux=True))(inputs)
            elif return_func_output and not return_jacobian:
                return functorch.vmap(functorch.jacrev(functorch.jacrev(aux_function(func), has_aux=True)))(inputs)
            elif return_func_output and return_jacobian:
                (hessian, (jacobian, value)) = functorch.vmap(
                    functorch.jacrev(aux_function_jac(functorch.jacrev(aux_function(func), has_aux=True)),
                                     has_aux=True))(inputs)
                return hessian, jacobian, value
    else:
        if not return_func_output and not return_jacobian:
            return functorch.vmap(functorch.jacrev(functorch.jacrev(func)))(inputs)
        elif not return_func_output and return_jacobian:
            return functorch.vmap(functorch.jacrev(aux_function_jac(functorch.jacrev(func)), has_aux=True))(inputs)
        elif return_func_output and not return_jacobian:
            return functorch.vmap(functorch.jacrev(functorch.jacrev(aux_function(func), has_aux=True)))(inputs)
        elif return_func_output and return_jacobian:
            (hessian, (jacobian, value)) = functorch.vmap(
                functorch.jacrev(aux_function_jac(functorch.jacrev(aux_function(func), has_aux=True)), has_aux=True))(
                inputs)
            return hessian, jacobian, value