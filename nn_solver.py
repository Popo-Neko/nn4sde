import torch
from torch import nn
import numpy as np
import pandas as pd
from equation import SDE
import abc
import yaml
from network import Net4Y
from time import time
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
import os
from sampler import Sampler
from torch.utils.data import DataLoader


class BSDESolver:
    def __init__(self, 
                 equation,
                 net,
                 configs):
        '''
        net: neural network
        equation: BSDE equation
        train_iter: number of iterations
        '''
        with open(configs, 'r') as f:
            config_data = yaml.safe_load(f)
            for key, value in config_data['train'].items():
                setattr(self, key, value)
        self.equation = equation
        self.net = net
        data = Sampler(self.equation, train=True)
        self.train_iter = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        
        # optimizer
        if self.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        else:
            raise ValueError('Optimizer not supported! Please check the configs file.')
    
    def loss(self, inputs):
        y_pred = self.net(inputs)
        y_mc = self.equation.terminal_condition(inputs[1])
        # l2 loss
        loss = torch.mean((y_pred - y_mc)**2)
        return loss

    def train(self):
        datetime = pd.Timestamp.now().strftime(r'%Y-%m-%d_%H-%M-%S')
        os.mkdir(f'logs/Experiment_{datetime}')
        # tensorboard writer
        writer = SummaryWriter(f'logs/Experiment_{datetime}')
        # create logging file with datetime and configs 
        logging.basicConfig(filename=f'logs/Experiment_{datetime}/{datetime}.log', level=logging.INFO)
        logging.info(f'Configs: {self.__dict__}')
        start_time = time()
        for epoch in tqdm(range(self.epochs), desc='Training'):
            self.optimizer.zero_grad()
            inputs = next(iter(self.train_iter))
            loss = self.loss(inputs)
            loss.backward()
            self.optimizer.step()
            logging.info(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}, Y0: {self.net.y_init.item()}')
            writer.add_scalar('Loss/train', loss.item(), epoch+1)
            writer.add_scalar('Y0/train', self.net.y_init.item(), epoch+1)
        end_time = time()
        logging.info(f'Training time: {end_time-start_time} seconds')
        writer.close()
            
        
    