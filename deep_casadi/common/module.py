import numpy as np
import casadi as cs

from deep_casadi.common.helper import is_casadi_type


class DeepCasadiModule:
    _input_size = None
    _output_size = None
    _sym_approx_params = {}

    @property
    def input_size(self):
        if self._input_size is not None:
            return self._input_size
        else:
            raise Exception('Input Size not known. Please set it in the constructor of your Module.')

    @input_size.setter
    def input_size(self, size):
        self._input_size = size

    @property
    def output_size(self):
        if self._output_size is not None:
            return self._output_size
        else:
            raise Exception('Output Size not known. Please set it in the constructor of your Module.')

    @output_size.setter
    def output_size(self, size):
        self._output_size = size

    def __call__(self, *args, **kwargs):
        if is_casadi_type(args[0]):
            try:
                return self.cs_forward(*args, **kwargs)
            except NotImplementedError:
                return self.forward(*args, **kwargs)
        else:
            return super().__call__(*args, **kwargs)

    def cs_forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_sym_params_list(self, recurse=True):
        raise NotImplementedError

    def get_params_list(self, recurse=True):
        raise NotImplementedError

    def params(self, symbolic=False, recurse=True, flat=False, approx=False, order=1, **kwargs):
        if symbolic:
            return self.sym_params(recurse=recurse, flat=flat, approx=approx, order=order)
        else:
            return self._params(recurse=recurse, flat=flat, approx=approx, order=order, **kwargs)

    def sym_params(self, recurse=True, flat=False, approx=False, order=1):
        if approx:
            sym_params_list = self.get_sym_approx_params_list(order)
        else:
            sym_params_list = self.get_sym_params_list(recurse=recurse)

        if not flat:
            return sym_params_list
        else:
            if len(sym_params_list) == 0:
                return cs.vertcat([])
            return cs.vcat([cs.reshape(mx, np.prod(mx.shape), 1) for mx in sym_params_list])

    def _params(self, recurse=True, flat=False, approx=False, a=None, order=1):
        if approx:
            assert a is not None, 'Approximation point a is missing.'
            params_list = self.get_approx_params_list(a, order=order)
        else:
            params_list = self.get_params_list(recurse=recurse)

        if not flat:
            return params_list
        else:
            if len(params_list) == 0:
                return np.array([])
            if approx and len(a.shape) > 1:
                return np.hstack([p.reshape(p.shape[0], -1) for p in params_list])
            return np.hstack([p.flatten() for p in params_list])

    def get_approx_params_list(self, a, order=1):
        raise NotImplementedError

    def get_sym_approx_params_list(self, order):

        if 'a' in self._sym_approx_params:
            a = self._sym_approx_params['a']
        else:
            a = cs.MX.sym('a', self.input_size, 1)
            self._sym_approx_params['a'] = a

        if 'f_a' in self._sym_approx_params:
            f_a = self._sym_approx_params['f_a']
        else:
            f_a = cs.MX.sym('f_a', self.output_size, 1)
            self._sym_approx_params['f_a'] = f_a

        if 'df_a' in self._sym_approx_params:
            df_a = self._sym_approx_params['df_a']
        else:
            df_a = cs.MX.sym('df_a', self.output_size, self.input_size)
            self._sym_approx_params['df_a'] = df_a


        ddf_as = []
        if order == 2:
            for i in range(self.output_size):
                if f'ddf_a_{i}' in self._sym_approx_params:
                    ddf_a_i = self._sym_approx_params[f'ddf_a_{i}']
                else:
                    ddf_a_i = cs.MX.sym(f'ddf_a_{i}', self.input_size, self.input_size)
                    self._sym_approx_params[f'ddf_a_{i}'] = ddf_a_i
                ddf_as.append(ddf_a_i)
        return [a, f_a, df_a] + ddf_as

    def approx(self, x: cs.MX, order=1):
        '''
        Approximation using first or second order Taylor Expansion
        '''
        approx_mx_params = self.sym_params(approx=True, order=order)
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

