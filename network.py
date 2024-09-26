import torch
from torch import nn
from typing import List, Tuple

class Net4Z(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: List[int],
                 output_dim: int):
        '''
        input_dim: number of input features should be the same as the number of assets N
        hidden_dim: a list of hidden layer dimensions
        output_dim: number of output features should be same as the dim of brownian motion D
        '''
        super(Net4Z, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = [nn.Linear(self.hidden_dim[i], self.hidden_dim[i+1]) for i in range(len(self.hidden_dim)-1)]
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim[0])
        self.output_layer = nn.Linear(self.hidden_dim[-1], self.output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.relu(self.output_layer(x))
        return x
    
    def init_weights(self):
        for layer in self.hidden_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
 
 
class Net4Y(Net4Z):
    def __init__(self, 
                 equation,
                 input_dim: int,
                 hidden_dim: List[int],
                 output_dim: int):
        '''
        input_dim: number of input features should be the same as the number of assets N
        hidden_dim: a list of hidden layer dimensions
        output_dim: number of output features should be same as the dim of brownian motion D
        '''
        super(Net4Y, self).__init__(input_dim, hidden_dim, output_dim)
        self.equation = equation
        # y_init torch.tensor (M, 1) all ones vector
        self.y_init = torch.ones(self.equation.M, 1)
        # z_init torch.tensor (M, D) normal random vector N(0, 0.025)
        self.z_init = torch.randn(self.equation.M, self.equation.D) * 0.05
        
        
    def forward(self, inputs):
        dw, x = inputs[0], inputs[1]
        y = self.y_init
        z = self.z_init
        dt = self.dt
        for t in range(self.equation.step):
            dw = dw[:, t, 1, :]
            y = y - self.equation.f(y) + 
        
    
    def init_weights(self):
        return super(Net4Y, self).init_weights()