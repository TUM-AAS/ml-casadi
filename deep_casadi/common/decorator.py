import casadi as cs

from deep_casadi.common.helper import is_casadi_type


def casadi(func_call, *args, **kwargs):
    casadi_call = None

    def func_wrapper(*args, **kwargs):
        test_arg = args[0] if len(args) > 0 else list(kwargs.values())[0]
        if type(test_arg) is list:
            test_arg = test_arg[0]
        if is_casadi_type(test_arg):
            if casadi_call is None:
                return getattr(cs, func_call.__name__)(*args, **kwargs)
            else:
                return casadi_call(*args, **kwargs)
        else:
            return func_call(*args, **kwargs)
    return func_wrapper
