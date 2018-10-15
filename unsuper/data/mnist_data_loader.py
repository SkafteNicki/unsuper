# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 12:16:57 2018

@author: Nicki
"""
#%%
import torch
from .mnist_data import MNIST

#%%
def mnist_data_loader(root, train=True, transform=None, target_transform=None, 
                 download=False, batch_size=128, classes=[0,1,2,3,4,5,6,7,8,9]):
    # Load dataset
    train = MNIST(root=root, train=True, transform=transform, download=download,
                  target_transform=target_transform, classes=classes)
    
    test = MNIST(root=root, train=False, transform=transform, download=download, 
                 target_transform=target_transform, classes=classes)
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    return trainloader, testloader

#%%
if __name__ == '__main__':
    trainloader, testloader = mnist_data_loader()