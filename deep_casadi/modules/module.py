import torch
import numpy as np
import casadi as cs


def jacobian_self(f, x):
    y = f(x)

    def get_vjp(v):
        return torch.autograd.grad(y, x, v, create_graph=True)[0]

    basis_vectors = torch.eye(y.shape[0])
    jacobian_vmap = torch.vmap(get_vjp)(basis_vectors)
    return jacobian_vmap


def batched_jacobian(f, x, create_graph=False, return_y=False):
    x.requires_grad = True
    y_o = []

    def f_sum(x):
        y = f(x)
        y_o.append(y)
        return torch.sum(y, axis=0)

    jacobian = torch.autograd.functional.jacobian(f_sum,
                                                  x,
                                                  vectorize=True,
                                                  create_graph=create_graph
                                                  ).moveaxis(-len(x.shape), 0)

    if return_y:
        return jacobian, y_o[0].detach()

    return jacobian


def batched_hessian(f, x, return_jacobian=False, return_y=False):
    additional_outputs = []

    def jacobian_func(x):
        out = batched_jacobian(f, x, create_graph=True, return_y=return_y)

        if return_y:
            additional_outputs.append(out[1])
            jacobian = out[0]
        else:
            jacobian = out

        if return_jacobian:
            additional_outputs.insert(0, jacobian.detach())

        return jacobian

    hessian = batched_jacobian(jacobian_func, x)
    if len(additional_outputs) > 0:
        return hessian, *additional_outputs
    return hessian


class Module(torch.nn.Module):
    _approx_mx_params = None
    _input_size = None
    _output_size = None

    @property
    def input_size(self):
        if self._input_size is not None:
            return self._input_size
        else:
            raise Exception('Input Size not set')

    @input_size.setter
    def input_size(self, size):
        self._input_size = size

    @property
    def output_size(self):
        if self._output_size is not None:
            return self._output_size
        else:
            raise Exception('Output Size not set')

    @output_size.setter
    def output_size(self, size):
        self._output_size = size

    def __call__(self, *args, **kwargs):
        if type(args[0]) is cs.MX:
            try:
                return self.cs_forward(*args, **kwargs)
            except NotImplementedError:
                return self.forward(*args, **kwargs)
        else:
            return super().__call__(*args, **kwargs)

    def cs_forward(self, input: cs.DM):
        raise NotImplementedError

    def mx_params(self, recurse=True, flat=False, approx=False, order=2):
        if approx:
            mx_params_list = self.approx_mx_params(order)
        else:
            def foo(module):
                for n, v in module._parameters.items():
                    casadi_att = module.__getattribute__('cs_' + n)
                    if type(casadi_att) is not cs.MX:
                        continue
                    yield n, casadi_att
            gen = self._named_members(
                foo,
                prefix='', recurse=recurse)
            mx_params_list = [mx for name, mx in gen]

        if not flat:
            return mx_params_list
        else:
            if len(mx_params_list) == 0:
                return cs.vertcat([])
            return cs.vcat([cs.reshape(mx, np.prod(mx.shape), 1) for mx in mx_params_list])

    def np_params(self, recurse=True, flat=False, approx=False, a=None, order=2):
        if approx:
            assert a is not None, 'Approximation point a is missing.'
            np_params_list = self.approx_np_params(a, order)
        else:
            np_params_list = [p.T.detach().numpy() for p in self.parameters(recurse)]

        if not flat:
            return np_params_list
        else:
            if len(np_params_list) == 0:
                return np.array([])
            if approx and len(a.shape) > 1:
                return np.hstack([p.reshape(p.shape[0], -1) for p in np_params_list])
            return np.hstack([p.flatten() for p in np_params_list])

    def approx_mx_params(self, order):
        return self.approx_mx_params_taylor(order)

    def approx_mx_params_taylor(self, order):
        if self._approx_mx_params is None:
            a = cs.MX.sym('a', self.input_size, 1)
            f_a = cs.MX.sym('f_a', self.output_size, 1)
            df_a = cs.MX.sym('df_a', self.output_size, self.input_size)
            ddf_as = []
            if order == 2:
                for i in range(self.output_size):
                    ddf_as.append(cs.MX.sym('ddf_a', self.input_size, self.input_size))
            self._approx_mx_params = [a, f_a, df_a] + ddf_as

        return self._approx_mx_params

    def approx_np_params(self, a: np.array, order):
        return self.approx_np_params_taylor(a, order)

    def approx_np_params_taylor(self, a: np.array, order):
        a_t = torch.tensor(a).float()
        if len(a_t.shape) == 1:
            a_t = a_t.unsqueeze(0)
        if order == 1:
            df_a, f_a = batched_jacobian(self, a_t, return_y=True)
            return [a, f_a.numpy(), df_a.transpose(-2, -1).numpy()]
        elif order == 2:
            ddf_a, df_a, f_a = batched_hessian(self, a_t, return_y=True, return_jacobian=True)
            return [a, f_a.numpy(), df_a.transpose(-2, -1).numpy()] + [ddf_a[:, i].transpose(-2, -1).numpy() for i in range(ddf_a.shape[1])]

    def approx_np_params_cs(self, a: cs.DM):
        dm_params_flat = self.np_params(flat=True)
        mx_params_flat = self.mx_params(flat=True)

        a_sym = cs.MX.sym('a_sym', a.shape)
        y = self(a_sym)
        f_f_a = cs.Function('f_a', [a_sym, mx_params_flat], [y])
        f_a = f_f_a(a, dm_params_flat)
        f_df_a = cs.Function('df_a', [a_sym, mx_params_flat], [cs.jacobian(y, a_sym)])
        df_a = f_df_a(a, dm_params_flat)
        ddf_as = []
        for i in range(y.shape[0]):
            f_ddf_a = cs.Function('ddf_a', [a_sym, mx_params_flat], [cs.hessian(y[i], a_sym)[0]])
            ddf_as.append(f_ddf_a(a, dm_params_flat).toarray())
        return [a.toarray(), f_a.toarray(), df_a.toarray()] + ddf_as

    def approx(self, x: cs.MX, order=2):
        return self.approx_taylor(x, order=order)

    def approx_taylor(self, x: cs.MX, order=2):
        '''
        Approximation using second order Taylor Expansion
        '''
        approx_mx_params = self.approx_mx_params(order)
        a = approx_mx_params[0]
        f_f_a = approx_mx_params[1]
        f_df_a = approx_mx_params[2]
        x_minus_a = x - a
        if order == 1:
            return (f_f_a
                    + cs.mtimes(f_df_a, x_minus_a))
        else:
            f_ddf_as = approx_mx_params[3:]
            return (f_f_a
                    + cs.mtimes(f_df_a, x_minus_a)
                    + 0.5 * cs.vcat(
                        [cs.mtimes(cs.transpose(x_minus_a), cs.mtimes(f_ddf_a, x_minus_a))
                         for f_ddf_a in f_ddf_as]))


    def approx_mx_params_two_point_taylor(self, order):
        assert order == 0
        if self._approx_mx_params is None:
            a = cs.MX.sym('a', self.input_size, 1)
            b = cs.MX.sym('b', self.input_size, 1)
            f_a = cs.MX.sym('f_a', self.output_size, 1)
            f_b = cs.MX.sym('f_b', self.output_size, 1)
            self._approx_mx_params = [a, b, f_a, f_b]

        return self._approx_mx_params

    '''
    def approx_np_params_two_point_taylor(self, a: np.array, b: np.array, order):
        assert order == 0
        a_t = torch.tensor(a).float()
        b_t = torch.tensor(b).float()
        if len(a_t.shape) == 1:
            a_t = a_t.unsqueeze(0)
        if len(b_t.shape) == 1:
            b_t = b_t.unsqueeze(0)

        f_a = self(a_t)
        f_b = self(b_t)

        return [a, b, f_a.detach().numpy(), f_b.detach().numpy()]
    
    def approx_two_point_taylor(self, x: cs.MX, order=2):
        assert order == 0, 'Only implemented zero order'
        approx_mx_params = self.approx_mx_params(order, algo='two_point_taylor')
        a = approx_mx_params[0]
        b = approx_mx_params[1]
        f_a = approx_mx_params[2]
        f_b = approx_mx_params[3]
        x_minus_a = x - a

        return f_a + ((f_b - f_a) / (b - a + 1e-6)) * x_minus_a
    '''