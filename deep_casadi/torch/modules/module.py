import torch

from deep_casadi.common import DeepCasadiModule
from deep_casadi.torch.autograd.functional import batched_jacobian, batched_hessian


class TorchDeepCasadiModule(DeepCasadiModule, torch.nn.Module):
    def get_approx_params_list(self, a, order=1):
        a_t = torch.tensor(a).float()
        if len(a_t.shape) == 1:
            a_t = a_t.unsqueeze(0)
        if order == 1:
            df_a, f_a = batched_jacobian(self, a_t, return_func_output=True)
            return [a, f_a.numpy(), df_a.transpose(-2, -1).numpy()]
        elif order == 2:
            ddf_a, df_a, f_a = batched_hessian(self, a_t, return_func_output=True, return_jacobian=True)
            return ([a, f_a.numpy(), df_a.transpose(-2, -1).numpy()]
                    + [ddf_a[:, i].transpose(-2, -1).numpy() for i in range(ddf_a.shape[1])])