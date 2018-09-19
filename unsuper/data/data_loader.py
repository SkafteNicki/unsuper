#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 07:49:31 2018

@author: nsde
"""

#%%
import torch
from torchvision import datasets
from ..helper.utility import get_dir
#%%
def mnist_train_loader(batch_size, transform):
    loc = get_dir(__file__) + '/MNIST/'
    return torch.utils.data.DataLoader(datasets.MNIST(loc, train=True, 
            download=True, transform=transform), batch_size=batch_size, shuffle=True)
            
            
#%%
def mnist_test_loader(batch_size, transform):
    loc = get_dir(__file__) + '/MNIST/'
    return torch.utils.data.DataLoader(datasets.MNIST(loc, train=False, 
            download=True, transform=transform), batch_size=batch_size, shuffle=True)
    