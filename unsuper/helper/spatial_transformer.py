#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:04:43 2018

@author: nsde
"""
#%%
import torch
from torch import nn
from torch.nn import functional as F
from .expm import torch_expm3x3 as expm

#%%
class STN_Affine(nn.Module):
    def __init__(self, input_shape):
        super(STN_Affine, self).__init__()
        self.input_shape = input_shape
        
    def forward(self, x, theta, inverse=False):
        theta = theta.view(-1, 2, 3)
        output_size = torch.Size([x.shape[0], *self.input_shape])
        grid = F.affine_grid(theta, output_size)
        x = F.grid_sample(x, grid)
        return x
    
#%%
class STN_AffineDiff(nn.Module):
    def __init__(self, input_shape):
        super(STN_AffineDiff, self).__init__()
        self.input_shape = input_shape
        
    def forward(self, x, theta, inverse=False):
        theta = theta.view(-1, 2, 3)
        theta = expm(theta)
        output_size = torch.Size([x.shape[0], *self.input_shape])
        grid = F.affine_grid(theta, output_size)
        x = F.grid_sample(x, grid)
        return x
        
#%%
if __name__ == '__main__':
    pass