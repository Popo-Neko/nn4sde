from typing import List, Tuple
import torch
import numpy as np


class Sampler:
    '''
    Use Feyman-Kac method as labels, check the nn method results.
    With the same dw, check the difference.
    equation: BSDE with terminal condition
    size: number of paths
    return dw, x, (y_mc if train = False else None) torch.tensor
    '''
    def __init__(self, 
                equation,
                batch_size,
                train=True):
        self.equation = equation
        self.train = train
        self.batch_size = batch_size
    
    def __len__(self):
        return self.batch_size
    
    def __getitem__(self, idx):
        _, self.x, self.dw = self.equation.simulate(num=self.batch_size)
        self.y_mc = self.equation.terminal_condition(self.x)
        self.x = torch.tensor(self.x, dtype=torch.float64)
        self.dw = torch.tensor(self.dw, dtype=torch.float64)
        self.y_mc = torch.tensor(self.y_mc, dtype=torch.float64)
        if self.train:
            return self.dw, self.x
        else:
            return self.dw, self.x, self.y_mc
    
    def __iter__(self):
        while True:
            yield self.__getitem__(0)

 

