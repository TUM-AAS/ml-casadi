from unittest import TestCase

import torch
import deep_casadi.torch as dc


class TestAutograd(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = lambda x: x ** 2
        self.B = 10
        self.N = 5
        self.x = torch.randn((self.B, self.N), requires_grad=True)

    def test_batched_jacobian(self):
        batched_jac = dc.autograd.functional.batched_jacobian(self.model, self.x)

        jac_list = []
        for i in range(self.B):
            jac_list.append(torch.autograd.functional.jacobian(self.model, self.x[i], vectorize=True))

        jac_it = torch.stack(jac_list)

        assert torch.allclose(jac_it, batched_jac)

    def test_batched_hessian(self):
        batched_hess = dc.autograd.functional.batched_hessian(self.model, self.x)

        hess_list = []
        for i in range(self.B):
            hess_list_list = []
            for j in range(self.N):
                hess_list_list.append(torch.autograd.functional.hessian(lambda x: self.model(x)[j], self.x[i],
                                                                        vectorize=True))

            hess_list.append(torch.stack(hess_list_list))

        hess_it = torch.stack(hess_list)

        assert torch.allclose(hess_it, batched_hess)
