import torch
from deep_casadi.torch.modules import TorchDeepCasadiModule
from deep_casadi.torch.modules.nn import Linear
from deep_casadi.torch.modules.nn import activation as activations


class MultiLayerPerceptron(TorchDeepCasadiModule):
    def __init__(self, input_size, hidden_size, output_size, n_hidden, activation=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = Linear(input_size, hidden_size)

        hidden = []
        for i in range(n_hidden):
            hidden.append((Linear(hidden_size, hidden_size)))
        self.hidden_layers = torch.nn.ModuleList(hidden)

        self.output_layer = Linear(hidden_size, output_size)

        if activation is None:
            self.act = lambda x: x
        elif type(activation) is str:
            self.act = getattr(activations, activation)()
        else:
            self.act = activation

    def forward(self, x):
        x = self.input_layer(x)
        x = self.act(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.act(x)
        y = self.output_layer(x)
        return y
