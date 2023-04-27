import pytest
import torch
from ml_casadi.torch.autograd.functorch import batched_jacobian, batched_hessian


class TestAutograd:
    @pytest.fixture
    def model(self):
        return lambda x: x ** 2

    @pytest.fixture(
        params=[1, 3, 5, 10, 20]
    )
    def batch_size(self, request):
        return request.param

    @pytest.fixture(
        params=[1, 3, 5, 10, 20]
    )
    def dim(self, request):
        return request.param

    @pytest.fixture
    def input(self, batch_size, dim):
        return torch.randn((batch_size, dim), requires_grad=True)

    def test_batched_jacobian(self, model, input, batch_size):
        x = input
        batched_jac = batched_jacobian(model, x)

        jac_list = []
        for i in range(batch_size):
            jac_list.append(torch.autograd.functional.jacobian(model, x[i], vectorize=True))

        jac_it = torch.stack(jac_list)

        assert torch.allclose(jac_it, batched_jac)

    def test_batched_hessian(self, model, input, batch_size, dim):
        x = input
        batched_hess = batched_hessian(model, x)

        hess_list = []
        for i in range(batch_size):
            hess_list_list = []
            for j in range(dim):
                hess_list_list.append(torch.autograd.functional.hessian(lambda x: model(x)[j], x[i],
                                                                        vectorize=True))

            hess_list.append(torch.stack(hess_list_list))

        hess_it = torch.stack(hess_list)

        assert torch.allclose(hess_it, batched_hess)
