import torch

from ml_casadi.common import MLCasadiModule
from ml_casadi.torch.autograd import batched_jacobian, batched_hessian


class TorchMLCasadiModule(MLCasadiModule, torch.nn.Module):
    def get_approx_params_list(self, a, order=1):
        device = next(self.parameters()).device
        a_t = torch.tensor(a).float().to(device)
        if len(a_t.shape) == 1:
            a_t = a_t.unsqueeze(0)
        if order == 1:
            df_a, f_a = batched_jacobian(self, a_t, return_func_output=True)
            return [a, f_a.cpu().numpy(), df_a.transpose(-2, -1).cpu().numpy()]
        elif order == 2:
            ddf_a, df_a, f_a = batched_hessian(self, a_t, return_func_output=True, return_jacobian=True)
            return ([a, f_a.cpu().numpy(), df_a.transpose(-2, -1).cpu().numpy()]
                    + [ddf_a[:, i].transpose(-2, -1).cpu().numpy() for i in range(ddf_a.shape[1])])


class TorchMLCasadiModuleWrapper(TorchMLCasadiModule, torch.nn.Module):
    def __init__(self, model: torch.nn.Module, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.wrapped_model = model

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)
