import numpy as np
import matplotlib.pyplot as plt
import abc
from tqdm import tqdm
import yaml
from typing import List, Tuple
import os
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sampler import Sampler

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
            for key, value in config_data['simulation'].items():
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
        self.step = self.N
        self.step, self.M = int(self.step), int(self.M)
        dt = self.T/self.step
        dW = np.sqrt(dt)*np.random.randn(self.M, self.step) # shape: [M, step]
        t = np.linspace(0, self.T, self.step+1) # shape: [step+1]
        return t, dW, dt
    
    def simulate(self, num=None, scheme="euler"):
        '''
        x0: initial value(shape: [M])
        T: terminal time(scalar)
        step: number of time steps(scalar)
        M: number of paths(scalar)
        N: number of assets(scalar)
        scheme: simulation scheme, "euler" or "milstein"
        return: time grid, simulated paths
        '''
        if num is not None:
            M = num
        else:
            M = self.M
        t, dW, dt = self.sample() 
        x = np.zeros((M, self.step+1))  # shape: [M, step+1]
        x[:, 0] = self.x0
        for i in tqdm(range(self.step), desc="Simulating Steps"):
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
                 volatility: np.ndarray,
                 configs: yaml,
                 x0 = None):
        '''
        drift: mu, shape=(N, 1), np.ndarray. N is the num of assets
        volatility: sigma, shape=(N, D), np.ndarray. D is the dimension of the brownian motion
        x0: manual input of initial value, shape=(N, 1), np.ndarray
        '''
        self.drift = drift
        self.volatility = volatility
        self.load_config(configs)
        self.D = self.volatility.shape[1]
        self.N = self.drift.shape[0]
        self.dt = self.T/self.step
        self.step, self.M= int(self.step), int(self.M) # number of time steps and paths
        # x0 shape = [N, 1]
        if x0 is None:
            self.x0 = np.array([self.x0 for i in range(self.N)]).reshape((self.N, 1))
        else:
            self.x0 = x0
        assert self.drift.shape[0] == self.volatility.shape[0], "mu and sigma must have the same num of assets"
        
    def mu(self, x, t=None):
        # x: (M, N) M = path
        # return: (M, N)
        M = x.shape[0]
        results = np.zeros((M, self.N))
        for i in range(self.N):
            results[:, i] = x[:, i]*self.drift[i]
        return results
    
    def sigma(self, x, t=None):
        # x: (M, N) M = path
        # return: (M, N, D)
        M = x.shape[0]
        results = np.zeros((M, self.N, self.D))
        for i in range(M):
            results[i] = x[i]*self.volatility
        return results
    
    def simulate(self, num=None, scheme="euler"):
        '''
        x0: initial value(shape: [M, N])
        T: terminal time(scalar)
        step: number of time steps(scalar)
        M: number of paths(scalar)
        N: number of assets(scalar)
        scheme: simulation scheme, "euler" or "milstein"
        for monte carlo simulation, num is None
        for training, num is the batch size
        return: time grid, simulated paths
        '''
        if num is None:
            M = self.M
        else:
            M = num
        # dW: brownian motion, shape: [M, step, N, D]
        dW = np.sqrt(self.dt)*np.random.randn(M, self.step, self.D) # shape: [M, step, D]
        # reshape dW to [M, step, N, D]
        dW = np.repeat(dW[:, :, np.newaxis, :], self.N, axis=2)
        
        # t: time grid, shape: [step+1] t0, t1, ..., tN
        t = np.linspace(0, self.T, self.step+1) # shape: [step+1]
        
        x = np.zeros((M, self.step+1, self.N))  # shape: [M, step+1, N]
        x[:, 0, :] = self.x0.T
        for i in tqdm(range(self.step), desc="Simulating Steps", disable= False if num is None else True):
            if scheme == "euler":
                x[:, i+1, :] = x[:, i, :] + self.mu(x[:, i, :], t[i])*self.dt + np.sum(self.sigma(x[:, i, :], t[i])*dW[:, i, :, :], axis=2)
            elif scheme == "milstein":
                x[:, i+1, :] = x[:, i, :] + self.mu(x[:, i, :], t[i])*self.dt + np.sum(self.sigma(x[:, i, :], t[i])*dW[:, i, :, :], axis=2) + 0.5*np.sum(self.sigma(x[:, i, :], t[i])*self.sigma(x[:, i, :], t[i])*(dW[:, i, :, :]**2 - self.dt), axis=2)
            else:
                raise ValueError("scheme must be 'euler' or 'milstein'")
        return t, x, dW
            

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


class EuropeanBasketOptionCall(GeometricBrownianMotionND):
    def __init__(self,
                 drift: np.ndarray,
                 volatility: np.ndarray,
                 strike: float,
                 risk_free_rate: float,
                 configs: yaml,
                 shares: np.ndarray,
                 x0 = None):
        '''
        drift: mu, shape=(N, 1), np.ndarray. N is the num of assets
        volatility: sigma, shape=(N, D), np.ndarray. D is the dimension of the brownian motion
        strike: scalar
        risk_free_rate: scalar
        configs: yaml file
        shares: shape=(N, 1), np.ndarray. the number of outstanding shares of each asset
        '''
        super().__init__(drift, volatility, configs, x0)
        self.strike = strike
        self.interest_rate = risk_free_rate
        self.shares = shares
        
    def f(self, y):
        return -y*self.interest_rate
    
    def terminal_condition(self, x):
        M = x.shape[0]
        market_value = x[:, -1, :] * self.shares.T
        total_value = market_value.sum(axis=1)

        weight = np.zeros((M, self.N)) if isinstance(x, np.ndarray) else torch.zeros((M, self.N), dtype=x.dtype)
        for i in range(self.N):
            weight[:, i] = market_value[:, i] / total_value

        y = (weight * x[:, -1, :]).sum(axis=1).reshape((M, 1))

        if isinstance(x, np.ndarray):
            return np.maximum(0, (y - self.strike).reshape((M, 1)))
        else:
            return torch.maximum(torch.tensor(0.0).to(y.device), (y - self.strike).reshape((M, 1)))


     
if  __name__ == "__main__":
    # # test for 1D
    # mu = 0.05  
    # sigma = 0.2
    # strike = 4
    # rf = 0.05

    # stock = GeometricBrownianMotion1D(mu, sigma, 'configs.yaml')
    # call = BlackSchloesCall1D(mu, sigma, strike, rf, configs='configs.yaml')
    
    # # y as call exact price
    # y_exact = call.exact_solution()

    # t, x, dw = stock.simulate(scheme='euler')
    
    # test for ND
    mu = np.array([0.2*i for i in range(1, 6)]).reshape((5, 1))
    sigma = np.array([[0.06*i for i in range(1, 6)] for j in range(5)])
    strike = 4
    rf = 0.05
    shares = np.array([1000*i for i in range(1, 6)]).reshape((5, 1))
    stock_index = EuropeanBasketOptionCall(mu, sigma, strike, rf, 'configs.yaml', shares)
    t, x, dw = stock_index.simulate(scheme='euler')
    y = stock_index.terminal_condition(x)
    
    
        
    
    
    
    
    
        