#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:23:49 2018

@author: nsde
"""

#%%
import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image

#%%
def perception_data_loader(root, transform=None, target_transform=None, 
                           download=False, batch_size=128, 
                           classes=[0,1,2,3,4,5,6,7,8,9], num_points=10000):
    # Load dataset
    train = PERCEPTION(root=root, train=True, transform=transform, download=download,
                       target_transform=target_transform, classes=classes, num_points=num_points)
    
    test = PERCEPTION(root=root, train=False, transform=transform, download=download, 
                      target_transform=target_transform, classes=classes, num_points=num_points)
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    return trainloader, testloader

#%%
class PERCEPTION(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 download=False, classes=[0,1,2,3,4,5,6,7,8,9], num_points = 20000):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        
        # Load file
        if self.train:
            self.data = torch.tensor(np.load('unsuper/data/PERCEPTION/training.npy')[:,0,:,:])
        else:
            self.data = torch.tensor(np.load('unsuper/data/PERCEPTION/testing.npy')[:,0,:,:])
        self.targets = torch.zeros(self.data.shape[0])
    
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
    
#%%
if __name__ == '__main__':
    dataset = PERCEPTION(' ')
    trainloader, testloader = perception_data_loader(root = ' ')
