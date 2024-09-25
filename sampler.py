from typing import List, Tuple
from torch.utils.data import Dataset
import torch
import numpy as np


class Sampler(Dataset):
    '''
    Use Feyman-Kac method as labels, check the nn method results.
    With the same dw, check the difference.
    equation: BSDE with terminal condition
    size: number of paths
    return dw, x, (y_mc if train = False else None) torch.tensor
    '''
    def __init__(self, 
                equation,
                train=True):
        self.equation = equation
        self.train = train
        _, self.x, self.dw = self.equation.simulate()
        self.y_mc = self.equation.terminal_condition(self.x)
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.dw = torch.tensor(self.dw, dtype=torch.float32)
        self.y_mc = torch.tensor(self.y_mc, dtype=torch.float32)
         
    def __len__(self):
        return self.equation.M
    
    def __getitem__(self, idx):
        if self.train:
            return self.dw[idx], self.x[idx]
        else:
            return self.dw[idx], self.x[idx], self.y_mc[idx]

