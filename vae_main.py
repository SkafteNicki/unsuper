# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 06:55:24 2018

@author: nsde
"""

#%%
from unsuper.data.data_loader import mnist_train_loader, mnist_test_loader
from unsuper import VAE

import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

#%%
def argparser():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--learning-rate', action="store", default=1e-4, type=float,
                        help='learning rate for optimizer')
    args = parser.parse_args()
    return args

#%%
if __name__ == '__main__':
    # Get input arguments
    args = argparser()
    
    # Get data loaders
    train_loader = mnist_train_loader(batch_size=128, transform=transforms.ToTensor())
    test_loader = mnist_test_loader(batch_size=128, transform=transforms.ToTensor())    
    input_shape = (1, 28, 28)
    
    # Define encoder and decoder
    latent_dim = 20
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.fc1 = nn.Linear(784, 400)
            self.fc2 = nn.Linear(400, latent_dim)
            self.fc3 = nn.Linear(400, latent_dim)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            h = self.relu(self.fc1(x))
            mu = self.fc2(h)
            logvar = self.fc3(h)
            return mu, logvar
        
    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.fc1 = nn.Linear(latent_dim, 400)
            self.fc2 = nn.Linear(400, 784)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            h  = self.relu(self.fc1(x))
            return torch.sigmoid(self.fc2(h))

    # Define model
    model = VAE(input_shape, latent_dim, Encoder, Decoder, device='cuda')
    
    # Train model
    model.train(train_loader, n_epochs=args.epochs, 
                learning_rate=args.learning_rate,
                testloader=test_loader)