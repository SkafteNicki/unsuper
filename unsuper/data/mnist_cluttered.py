# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:06:48 2018

@author: Nicki
"""
#%%
import numpy as np
import torch

#%%
def mnist_cluttered_data_loader(batch_size):
    mnist_cluttered = np.load('mnist_sequence1_sample_5distortions5x5.npz')
    X_train = torch.tensor(mnist_cluttered['X_train']).reshape(-1, 1, 40, 40)
    y_train = torch.tensor(mnist_cluttered['y_train'])
    X_test = torch.tensor(mnist_cluttered['X_test']).reshape(-1, 1, 40, 40)
    y_test = torch.tensor(mnist_cluttered['y_test'])
    
    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test, y_test)
    
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffel=False)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffel=False)
    return trainloader, testloader
    