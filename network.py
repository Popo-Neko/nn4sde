import torch
from torch import nn
from typing import List, Tuple
import yaml

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
        self.hidden_layers = []
        self.hidden_layers.append(nn.BatchNorm1d(self.hidden_dim[0]))
        for i in range(len(self.hidden_dim)-1):
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i+1]))
            self.hidden_layers.append(nn.BatchNorm1d(self.hidden_dim[i+1]))
            self.hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim[0])
        self.output_layer = nn.Linear(self.hidden_dim[-1], self.output_dim)
        self.relu = nn.ReLU()
        
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
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
                 configs: yaml):
        '''
        input_dim: number of input features should be the same as the number of assets N
        hidden_dim: a list of hidden layer dimensions
        output_dim: number of output features should be same as the dim of brownian motion D
        '''
        self.equation = equation
        self.input_dim = self.equation.N
        self.output_dim = self.equation.D
        
        with open(configs, 'r') as f:
            config_data = yaml.safe_load(f)
            for key, value in config_data['net'].items():
                setattr(self, key, value)
                
        super(Net4Y, self).__init__(self.input_dim, self.hidden_dim, self.output_dim)
        self.net = Net4Z(self.input_dim, self.hidden_dim, self.output_dim)
        
        self.y_init = nn.Parameter(torch.ones(1))
        self.z_init = nn.Parameter(torch.ones(1, self.equation.D)*0.1)
        
    def forward(self, inputs):
        dw, x = inputs[0], inputs[1]
        y = torch.ones(x.shape[0], 1) * self.y_init # y_init torch.tensor (M, 1) all ones vector
        z = torch.ones(x.shape[0], self.equation.D) * self.z_init # z_init torch.tensor (M, D) 
        dt = self.equation.dt
        for t in range(self.equation.step-1):
            y = y - self.equation.f(y)*dt + torch.sum(z*dw[:, t, 1, :], dim=1, keepdim=True)
            z = self.net(x[:, t, :])
        # terminal condition
        y = y - self.equation.f(y)*dt + torch.sum(z*dw[:, -1, 1, :], dim=1, keepdim=True)
        # take max(0, y)
        y = torch.max(y, torch.zeros_like(y))
        return y    
    
    def init_weights(self):
        return super(Net4Y, self).init_weights()