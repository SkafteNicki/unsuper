#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:56:53 2018

@author: nsde
"""
#%%
import os
import torch
from torch import nn
import numpy as np

#%%
def get_dir(file):
    """ Get the folder of specified file """
    return os.path.dirname(os.path.realpath(file))

#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#%%
def model_summary(model):
    print(40*"=" + " Model Summary " + 40*"=")
    print(model)
    print('Number of parameters:', count_parameters(model))
    print(95*"=")

#%%
class CenterCrop(nn.Module):
    def __init__(self, h, w):
        super(CenterCrop, self).__init__()
        self.h = h
        self.w = w
        
    def forward(self, x):
        h, w = x.shape[2:]
        x1 = int(round((h - self.h) / 2.))
        y1 = int(round((w - self.w) / 2.))
        out = x[:,:,x1:x1+self.h,y1:y1+self.w]
        return out
    
#%%
def affine_decompose(A):
    sx = (A[:,0,0].pow(2) + A[:,1,0].pow(2)).sqrt()
    sy = (A[:,1,1] * A[:,0,0] - A[:,0,1] * A[:,1,0]) / sx
    m = (A[:,0,1] * A[:,0,0] + A[:,1,0] * A[:,1,1]) / (A[:,1,1] * A[:,0,0] - A[:,0,1] * A[:,1,0])
    theta = torch.atan2(A[:,1,0] / sx, A[:,0,0] / sx)
    tx = A[:, 0, 2]
    ty = A[:, 1, 2]
    return sx, sy, m, theta, tx, ty

#%%
def log_p_multi_normal(x, means):
    constant = 1.0/(2*np.pi)
    return (means - x).norm(p=2, dim=1).mul(-0.5).exp().mul(constant).mean().log()

#%%
if __name__ == '__main__':
    pass