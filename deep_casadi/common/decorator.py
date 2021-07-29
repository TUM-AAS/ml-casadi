from functools import partial

import casadi as cs

from deep_casadi.common.helper import is_casadi_type


class casadi:
    def __init__(self, func_call):
        self.func_call = func_call
        self.explicit_casadi_call = None

    def __get__(self, obj, objtype=None):
        return partial(self.__call__, obj)

    def explicit(self, func_call):
        self.explicit_casadi_call = func_call

    def __call__(self, *args, **kwargs):
        test_args = list(args) + list(kwargs.values())
        is_casadi = False
        for arg in test_args:
            if is_casadi_type(arg):
                is_casadi = True
            if type(arg) is list:
                if is_casadi_type(arg[0]):
                    is_casadi = True
            if is_casadi:
                break

        if is_casadi:
            if self.explicit_casadi_call is None:
                return getattr(cs, self.func_call.__name__)(*args, **kwargs)
            else:
                return self.explicit_casadi_call(*args, **kwargs)
        else:
            return self.func_call(*args, **kwargs)
