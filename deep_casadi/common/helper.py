import casadi as cs


def is_casadi_type(x):
    x_type = type(x)

    if x_type is cs.MX or x_type is cs.SX or x_type is cs.DM:
        return True

    return False


def is_symbolic_casadi_type(x):
    x_type = type(x)

    if x_type is cs.MX or x_type is cs.SX:
        return True

    return False
