#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:18:28 2018

@author: nsde
"""
#%%
import torch
import argparse
from torchvision import datasets, transforms
from unsuper.data.mnist_data_loader import MNIST

#%%
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
    parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--classes','--list', nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='classes to train on')
    args = parser.parse_args()
    return args

#%%
if __name__ == '__main__':
    # Input arguments
    args = argparser()
    img_size = (args.channels, args.img_size, args.img_size)
    
    # Load data
    train = MNIST(root='unsuper/data', transform=transforms.ToTensor(), 
                           download=True, classes=args.classes)
    test = MNIST(root='unsuper/data', train=False, transform=transforms.ToTensor(), 
                          download=True, classes=args.classes)
    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size)
    testloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size)
    
