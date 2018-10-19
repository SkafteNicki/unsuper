#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:56:53 2018

@author: nsde
"""
#%%
import os
from torch import nn

#%%
def get_dir(file):
    """ Get the folder of specified file """
    return os.path.dirname(os.path.realpath(file))

#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        return x[:,:,x1:x1+self.h,y1:y1+self.w]

#%%
if __name__ == '__main__':
    pass