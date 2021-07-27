from typing import Callable

import torch


def batched_jacobian(func: Callable, inputs: torch.Tensor, create_graph=False, return_func_output=False):
    r"""Function that computes batches of the Jacobian of a given function and a batch of inputs.
    :param func: a Python function that takes Tensor inputs and returns a tuple of Tensors or a Tensor.
    :param inputs: inputs to the function ``func``. First dimension is treated as batch dimension
    :param create_graph: If ``True``, the Jacobian will be computed in a differentiable manner.
    :param return_func_output: If ``True``, the function output will be returned.
    :return: Jacobian
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
    r"""Function that computes batches of the Hessian of a given function and a batch of inputs.

    :param func: a Python function that takes Tensor inputs and returns a tuple of Tensors or a Tensor.
    :param inputs: inputs to the function ``func``. First dimension is treated as batch dimension
    :param create_graph: If ``True``, the Hessian will be computed in a differentiable manner.
    :param return_jacobian: If ``True``, the Jacobian will be returned.
    :param return_func_output: If ``True``, the function output will be returned.
    :return: Hessian
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
        return hessian if create_graph else hessian.detach(), *additional_outputs

    return hessian if create_graph else hessian.detach()
