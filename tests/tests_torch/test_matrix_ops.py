import numpy as np
import pytest

import torch
import casadi as cs
import ml_casadi.torch as mc


class TestMatrixOps:
    @pytest.fixture(
        params=[(1, 3), (2, 3), (3, 1)]
    )
    def tensors(self, request):
        size = request.param
        sym_a = cs.MX.sym('a', size[0], size[1])
        sym_b = cs.MX.sym('a', size[0], size[1])

        tensor_a = torch.rand(size)
        tensor_b = torch.rand(size)

        dm_a = cs.DM(tensor_a.numpy())
        dm_b = cs.DM(tensor_b.numpy())

        return sym_a, sym_b, tensor_a, tensor_b, dm_a, dm_b

    def test_vcat(self, tensors):
        sym_a, sym_b, tensor_a, tensor_b, dm_a, dm_b = tensors
        func = mc.vcat
        assert func([sym_a, sym_b]).shape == func([tensor_a, tensor_b]).shape
        assert func([dm_a, dm_b]).shape == func([tensor_a, tensor_b]).shape
        assert np.allclose(func([dm_a, dm_b]).toarray(), func([tensor_a, tensor_b]).numpy())

    def test_vertcat(self, tensors):
        sym_a, sym_b, tensor_a, tensor_b, dm_a, dm_b = tensors
        func = mc.vertcat
        assert func(sym_a, sym_b).shape == func(tensor_a, tensor_b).shape
        assert func(dm_a, dm_b).shape == func(tensor_a, tensor_b).shape
        assert np.allclose(func(dm_a, dm_b).toarray(), func(tensor_a, tensor_b).numpy())

    def test_hcat(self, tensors):
        sym_a, sym_b, tensor_a, tensor_b, dm_a, dm_b = tensors
        func = mc.hcat
        assert func([sym_a, sym_b]).shape == func([tensor_a, tensor_b]).shape
        assert func([dm_a, dm_b]).shape == func([tensor_a, tensor_b]).shape
        assert np.allclose(func([dm_a, dm_b]).toarray(), func([tensor_a, tensor_b]).numpy())

    def test_horzcat(self, tensors):
        sym_a, sym_b, tensor_a, tensor_b, dm_a, dm_b = tensors
        func = mc.horzcat
        assert func(sym_a, sym_b).shape == func(tensor_a, tensor_b).shape
        assert func(dm_a, dm_b).shape == func(tensor_a, tensor_b).shape
        assert np.allclose(func(dm_a, dm_b).toarray(), func(tensor_a, tensor_b).numpy())
