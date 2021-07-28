from unittest import TestCase

import numpy as np

import torch
import casadi as cs
import deep_casadi.torch as dc


class TestMatrixOps(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        size = (2, 3)
        self.sym_a = cs.MX.sym('a', size[0], size[1])
        self.sym_b = cs.MX.sym('a', size[0], size[1])

        self.tensor_a = torch.rand(size)
        self.tensor_b = torch.rand(size)

        self.dm_a = cs.DM(self.tensor_a.numpy())
        self.dm_b = cs.DM(self.tensor_b.numpy())

    def test_vcat(self):
        func = dc.vcat
        assert func([self.sym_a, self.sym_b]).shape == func([self.tensor_a, self.tensor_b]).shape
        assert func([self.dm_a, self.dm_b]).shape == func([self.tensor_a, self.tensor_b]).shape
        assert np.allclose(func([self.dm_a, self.dm_b]).toarray(), func([self.tensor_a, self.tensor_b]).numpy())

    def test_vertcat(self):
        func = dc.vertcat
        assert func(self.sym_a, self.sym_b).shape == func(self.tensor_a, self.tensor_b).shape
        assert func(self.dm_a, self.dm_b).shape == func(self.tensor_a, self.tensor_b).shape
        assert np.allclose(func(self.dm_a, self.dm_b).toarray(), func(self.tensor_a, self.tensor_b).numpy())

    def test_hcat(self):
        func = dc.hcat
        assert func([self.sym_a, self.sym_b]).shape == func([self.tensor_a, self.tensor_b]).shape
        assert func([self.dm_a, self.dm_b]).shape == func([self.tensor_a, self.tensor_b]).shape
        assert np.allclose(func([self.dm_a, self.dm_b]).toarray(), func([self.tensor_a, self.tensor_b]).numpy())

    def test_horzcat(self):
        func = dc.horzcat
        assert func(self.sym_a, self.sym_b).shape == func(self.tensor_a, self.tensor_b).shape
        assert func(self.dm_a, self.dm_b).shape == func(self.tensor_a, self.tensor_b).shape
        assert np.allclose(func(self.dm_a, self.dm_b).toarray(), func(self.tensor_a, self.tensor_b).numpy())
