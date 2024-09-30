from sampler import Sampler
from network import Net4Y
from nn_solver import BSDESolver
from equation import EuropeanBasketOptionCall
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':
    # configs
    mu = np.array([0.02 for i in range(1, 6)]).reshape((5, 1))
    sigma = np.array([[0.2 for i in range(1, 6)] for j in range(5)])
    strike = 1
    rf = 0.05
    shares = np.array([1000*i for i in range(1, 6)]).reshape((5, 1))
    config = "configs/configs.yaml"
    equation = EuropeanBasketOptionCall(mu, sigma, strike, rf, config, shares)
    net = Net4Y(equation, config)
    solver = BSDESolver(equation, net, config)
    solver.train()
    t, x, dw = equation.simulate(scheme='euler')
    y_mc = equation.terminal_condition(x)
    x = torch.tensor(x, dtype=torch.float32)
    dw = torch.tensor(dw, dtype=torch.float32)
    y_nn = net((dw, x)).detach().numpy()
    # calculate the mean squared error
    mse = ((y_nn - y_mc)**2).mean()
    print(f'Mean Squared Error: {mse}')
        