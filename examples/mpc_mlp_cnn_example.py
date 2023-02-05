import casadi as cs
import numpy as np
import torch
import torchvision.models as models
import ml_casadi.torch as mc
from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

MODEL_ARCH = 'MLP'  # On of 'MLP' or 'CNNRESNET'
# MLP: Dense Network of 32 layers 256 neurons each
# CNNRESNET: ResNet model with 18 Convolutional layers

USE_GPU = False


class CNNResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.res_net = models.resnet18()

        self.linear = torch.nn.Linear(1000, 2, bias=False)

        # Model is not trained -- setting output to zero
        with torch.no_grad():
            self.linear.weight.fill_(0.)

    def forward(self, x):
        # Tile input such that it fits the expected ResNet Input
        x = x[:, None, None, :].repeat(1, 3, 64, 32)
        return self.linear(self.res_net(x))


class MLP(mc.nn.MultiLayerPerceptron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Model is not trained -- setting output to zero
        with torch.no_grad():
            self.output_layer.bias.fill_(0.)
            self.output_layer.weight.fill_(0.)


class DoubleIntegratorWithLearnedDynamics:
    def __init__(self, learned_dyn):
        self.learned_dyn = learned_dyn

    def model(self):
        s = cs.MX.sym('s', 1)
        s_dot = cs.MX.sym('s_dot', 1)
        s_dot_dot = cs.MX.sym('s_dot_dot', 1)
        u = cs.MX.sym('u', 1)
        x = cs.vertcat(s, s_dot)
        x_dot = cs.vertcat(s_dot, s_dot_dot)

        res_model = self.learned_dyn.approx(x)
        p = self.learned_dyn.sym_approx_params(order=1, flat=True)
        parameter_values = self.learned_dyn.approx_params(np.array([0, 0]), flat=True, order=1)

        f_expl = cs.vertcat(
            s_dot,
            u
        ) + res_model

        x_start = np.zeros((2))

        # store to struct
        model = cs.types.SimpleNamespace()
        model.x = x
        model.xdot = x_dot
        model.u = u
        model.z = cs.vertcat([])
        model.p = p
        model.parameter_values = parameter_values
        model.f_expl = f_expl
        model.x_start = x_start
        model.constraints = cs.vertcat([])
        model.name = "wr"

        return model


class MPC:
    def __init__(self, model, N):
        self.N = N
        self.model = model

    @property
    def simulator(self):
        return AcadosSimSolver(self.sim())

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())

    def sim(self):
        model = self.model

        t_horizon = 1.
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        model_ac.p = model.p

        # Dimensions
        nx = 2
        nu = 1
        ny = 1

        # Create OCP object to formulate the optimization
        sim = AcadosSim()
        sim.model = model_ac
        sim.dims.N = N
        sim.dims.nx = nx
        sim.dims.nu = nu
        sim.dims.ny = ny
        sim.solver_options.tf = t_horizon

        # Solver options
        sim.solver_options.Tsim = 1./ 10.
        sim.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        sim.solver_options.hessian_approx = 'GAUSS_NEWTON'
        sim.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.print_level = 0
        sim.solver_options.nlp_solver_type = 'SQP_RTI'

        return sim

    def ocp(self):
        model = self.model

        t_horizon = 1.
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        model_ac.p = model.p

        # Dimensions
        nx = 2
        nu = 1
        ny = 1

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        # Initialize cost function
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        ocp.cost.W = np.array([[1.]])

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[0, 0] = 1.
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vz = np.array([[]])
        ocp.cost.Vx_e = np.zeros((ny, nx))
        ocp.cost.W_e = np.array([[0.]])
        ocp.cost.yref_e = np.array([0.])

        # Initial reference trajectory (will be overwritten)
        ocp.cost.yref = np.zeros(1)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = model.x_start

        # Set constraints
        a_max = 10
        ocp.constraints.lbu = np.array([-a_max])
        ocp.constraints.ubu = np.array([a_max])
        ocp.constraints.idxbu = np.array([0])

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        ocp.parameter_values = model.parameter_values

        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.name = model.name
        return model_ac


def run():
    N = 10
    if MODEL_ARCH == 'MLP':
        learned_dyn_model = MLP(2, 256, 2, 32, 'Tanh')
    elif MODEL_ARCH == 'CNNRESNET':
        learned_dyn_model = mc.TorchMLCasadiModuleWrapper(CNNResNet(), input_size=2, output_size=2)

    if USE_GPU:
        print('Moving Model to GPU')
        learned_dyn_model = learned_dyn_model.to('cuda:0')
        print('Model is on GPU')

    model = DoubleIntegratorWithLearnedDynamics(learned_dyn_model)
    solver = MPC(model=model.model(), N=N).solver

    print('Warming up model...')
    x_l = []
    for i in range(N):
        x_l.append(solver.get(i, "x"))
    for i in range(20):
        learned_dyn_model.approx_params(np.stack(x_l, axis=0), flat=True)
    print('Warmed up!')

    x = []
    x_ref = []
    ts = 1. / N
    xt = np.array([1., 0.])
    opt_times = []

    for i in range(50):
        now = time.time()
        t = np.linspace(i * ts, i * ts + 1., 10)
        yref = np.sin(0.5 * t + np.pi / 2)
        x_ref.append(yref[0])
        for t, ref in enumerate(yref):
            solver.set(t, "yref", ref)
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)
        solver.solve()
        xt = solver.get(1, "x")
        x.append(xt)

        x_l = []
        for i in range(N):
            x_l.append(solver.get(i, "x"))
        params = learned_dyn_model.approx_params(np.stack(x_l, axis=0), flat=True)
        for i in range(N):
            solver.set(i, "p", params[i])

        elapsed = time.time() - now
        opt_times.append(elapsed)

    print(f'Mean iteration time with CNN ResNet Model: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')


if __name__ == '__main__':
    run()