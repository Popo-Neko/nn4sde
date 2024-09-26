from sampler import Sampler
from network import Net4Y
from nn_solver import BSDESolver
from equation import EuropeanBasketOptionCall
import numpy as np

if __name__ == '__main__':
    # configs
    mu = np.array([0.2*i for i in range(1, 6)]).reshape((5, 1))
    sigma = np.array([[0.06*i for i in range(1, 6)] for j in range(5)])
    strike = 4
    rf = 0.05
    shares = np.array([1000*i for i in range(1, 6)]).reshape((5, 1))
    config = "configs/configs.yaml"
    equation = EuropeanBasketOptionCall(mu, sigma, strike, rf, config, shares)
    net = Net4Y(equation, config)
    solver = BSDESolver(equation, net, config)
    solver.train()