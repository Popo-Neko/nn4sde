import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import matplotlib.pyplot as plt
import abc
from tqdm import tqdm
import yaml
from typing import List, Tuple
import os
from scipy.stats import norm


# define a abstract SDE
class SDE(abc.ABC):
    def __init__(self):
        pass
    @abc.abstractmethod
    def mu(self, x, t):
        pass

    @abc.abstractmethod
    def sigma(self, x, t):
        pass
    
    def load_config(self, configs):
        with open(configs, 'r') as f:
            config_data = yaml.safe_load(f)
            for key, value in config_data.items():
                setattr(self, key, value)


class GeometricBrownianMotion1D(SDE):
    def __init__(self, 
                 drift,
                 volatility,
                 configs):
        self.dirft = drift
        self.volatility = volatility
        self.load_config(configs)
    
    def mu(self, x, t=None):
        return self.dirft*x
    
    def sigma(self, x, t=None):
        return self.volatility*x
    
    def sample(self):
        self.N, self.M = int(self.N), int(self.M)
        dt = self.T/self.N
        dW = np.sqrt(dt)*np.random.randn(self.M, self.N) # shape: [M, N]
        t = np.linspace(0, self.T, self.N+1) # shape: [N+1]
        return t, dW, dt
    
    def simulate(self, scheme="euler"):
        '''
        x0: initial value(shape: [M])
        T: terminal time(scalar)
        N: number of time steps(scalar)
        M: number of paths(scalar)
        scheme: simulation scheme, "euler" or "milstein"
        return: time grid, simulated paths
        '''
        t, dW, dt = self.sample() 
        x = np.zeros((self.M, self.N+1))  # shape: [M, N+1]
        x[:, 0] = self.x0
        for i in tqdm(range(self.N), desc="Simulating Steps"):
            if scheme == "euler":
                x[:, i+1] = x[:, i] + self.mu(x[:, i], t[i])*dt + self.sigma(x[:, i], t[i])*dW[:, i]
            elif scheme == "milstein":
                x[:, i+1] = x[:, i] + self.mu(x[:, i], t[i])*dt + self.sigma(x[:, i], t[i])*dW[:, i] + 0.5*self.sigma(x[:, i], t[i])*self.sigma(x[:, i], t[i])*(dW[:, i]**2 - dt)
            else:
                raise ValueError("scheme must be 'euler' or 'milstein'")
        return t, x, dW 
    
    def exact_solution(self, dW: np.ndarray):
        # sum the np array on the axis of 1
        W = np.sum(dW, axis=1) # shape: [M]
        return self.x0*np.exp((self.dirft - 0.5*self.volatility**2)*self.T + self.volatility*W) # use broadcasting: scalar + vector -> vector.shape


class GeometricBrownianMotionND(SDE):
    def __init__(self,
                 drift: np.ndarray,
                 volatility: np.ndarray):
        '''
        drift: mu, shape=(N, 1), np.ndarray. N is the num of assets(return vector) 
        volatility: sigma, shape=(N, D), np.ndarray. D is the num of dw dimensions(diffusion matrix)
        '''
        self.drift = drift
        self.volatility = volatility
        self.N = self.dirft.shape[0]
        self.D = self.volatility.shape[1]
        assert self.dirft.shape[0] == self.volatility.shape[0], "drift and volatility must have the same num of assets"
    
    def mu(self, x, t=None):
        # x: (M, 1) M = path
        # return: (M, N)
        return np.matmul(x, self.dirft.T)
    
    def sigma(self, x, t=None):
        # x: (M, 1) M = path
        # return: (M, N, D)
        results = np.zeros((x.shape[0], self.N, self.D))
        for i in range(x.shape[0]):
            results[i] = x[i]*self.volatility
        return results


class BlackSchloesCall1D(GeometricBrownianMotion1D):
    def __init__(self, 
                 drift,
                 volatility,
                 strike,
                 risk_free_rate, configs):
        super().__init__(drift, volatility, configs)
        self.strike = strike
        self.interest_rate = risk_free_rate
        self.d1 = (np.log(self.x0/self.strike) + (self.dirft + 0.5*self.volatility**2)*self.T)/(self.volatility*np.sqrt(self.T))
        self.d2 = self.d1 - self.volatility*np.sqrt(self.T)

    def f(self, y):
        return  -y*self.interst_rate
    
    def terminal_condition(self, y):
        return max(0, y - self.strike)
    
    def exact_solution(self):
        return self.x0*norm.cdf(self.d1) - self.strike*np.exp(-self.interest_rate*self.T)*norm.cdf(self.d2)


class BlackSchloesCallND(GeometricBrownianMotionND):
    def __init__(self, 
                 drift,
                 volatility,
                 strike,
                 risk_free_rate, configs):
        super().__init__(drift, volatility, configs)
        self.strike = strike
        self.interest_rate = risk_free_rate
        self.d1 = (np.log(self.x0/self.strike) + (self.dirft + 0.5*self.volatility**2)*self.T)/(self.volatility*np.sqrt(self.T))
        self.d2 = self.d1 - self.volatility*np.sqrt(self.T)

    def f(self, y):
        return  -y*self.interst_rate
    
    def terminal_condition(self, y):
        return max(0, y - self.strike)
    
    def exact_solution(self):
        return self.x0*norm.cdf(self.d1) - self.strike*np.exp(-self.interest_rate*self.T)*norm.cdf(self.d2)
  
    
if  __name__ == "__main__":
    l1 = []
    for i in tqdm(range(100)):
        mu = 0.05  
        sigma = 0.2
        strike = 1.25
        rf = 0.05
        stock = GeometricBrownianMotion1D(mu, sigma, 'configs.yaml')
        call = BlackSchloesCall1D(mu, sigma, strike, rf, configs='configs.yaml')
        # y as call exact price
        y_exact = call.exact_solution()

        t, x, dw = stock.simulate(scheme='euler')
        # exact = stock.exact_solution(dw)
        # for i in range(5):
        #     plt.plot(t, x[i, :].reshape((call.N+1)), label=f'path {i}')
        #     plt.scatter(call.T, exact[i], label=f'exact {i}')
        # plt.title(f'Approximation Path and Exact Solution \n stock price(mu={mu}, sigma={sigma})')
        # plt.legend()
        # plt.show()

        x_T = x[:, -1]
        y_T = []
        for i in x_T:
            _ = call.terminal_condition(i)
            y_T.append(_)
        y_T = np.array(y_T)
        y_mc = y_T.mean()
        T = call.T
        l1.append(np.abs(y_exact - y_mc*np.exp(-rf*T))/y_exact)
    print(np.array(l1).mean(), np.array(l1).std())
    
    
        