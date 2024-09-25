import torch
from torch import nn
import numpy as np
import pandas as pd
from equation import SDE
import abc
import yaml


class BSDESolver:
    def __init__(self, 
                 net,
                 equation,
                 train_iter):
        '''
        net: neural network
        equation: BSDE equation
        train_iter: number of iterations
        '''
        self.equation = equation
        self.net = net
        self.train_iter = train_iter
    