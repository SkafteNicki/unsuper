# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:15:20 2018

@author: nsde
"""

#%%
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

#%%
class MLP_Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim=32, h_size=[256, 128, 64]):
        super(MLP_Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.flat_dim = np.prod(input_shape)
        self.fc_layers = [nn.Linear(self.flat_dim, h_size[0])]
        for i in range(1, len(h_size)):
            self.fc_layers.append(nn.Linear(h_size[i-1], h_size[i]))
        self.fc_layers.append(nn.Linear(h_size[-1], self.latent_dim))
        self.fc_layers.append(nn.Linear(h_size[-1], self.latent_dim))
        self.activation = nn.LeakyReLU(0.1)
        self.paramlist = nn.ParameterList()
        for l in self.fc_layers:
            for p in l.parameters():
                self.paramlist.append(p)
        
    def forward(self, x):
        h = x.view(-1, self.flat_dim)
        for i in range(len(self.fc_layers)-2):
            h = self.activation(self.fc_layers[i](h))
        mu = self.fc_layers[-2](h)
        logvar = F.softplus(self.fc_layers[-1](h))
        return mu, logvar
    
#%%     
class MLP_Decoder(nn.Module):
    def __init__(self, output_shape, latent_dim=32, h_size=[64, 128, 256], 
                 end_activation=torch.sigmoid):
        super(MLP_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.flat_dim = np.prod(output_shape)
        self.activation = nn.LeakyReLU(0.1)
        self.end_activation = end_activation
        self.fc_layers = [nn.Linear(self.latent_dim, h_size[0])]
        for i in range(1, len(h_size)):
            self.fc_layers.append(nn.Linear(h_size[i-1], h_size[i]))
        self.fc_layers.append(nn.Linear(h_size[-1], self.flat_dim))
        self.paramlist = nn.ParameterList()
        for l in self.fc_layers:
            for p in l.parameters():
                self.paramlist.append(p)
        
    def forward(self, x):
        h  = self.activation(self.fc_layers[0](x))
        for i in range(1, len(self.fc_layers)-1):
            h = self.activation(self.fc_layers[i](h))
        out = self.end_activation(self.fc_layers[-1](h))
        return out.view(-1, *self.output_shape)
    
#%%
class Conv_Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim=32):
        super(Conv_Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 
        self.drop = nn.Dropout(0.8)        
        self.fc1 = nn.Linear(64*7*7, latent_dim)
        self.fc2 = nn.Linear(64*7*7, latent_dim)
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        n = x.shape[0]
        h1 = self.drop(self.activation(self.conv1(x)))
        h2 = self.drop(self.activation(self.conv2(h1)))
        h3 = self.drop(self.activation(self.conv3(h2)))
        mu = self.fc1(h3.view(n, -1))
        logvar = F.softplus(self.fc2(h3.view(n, -1)))
        return mu, logvar
    
#%%
class Conv_Decoder(nn.Module):
    def __init__(self, output_shape, latent_dim=32, end_activation=torch.sigmoid):
        super(Conv_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.drop = nn.Dropout(0.8)
        self.fc1 = nn.Linear(latent_dim, 7*7*1)
        self.fc2 = nn.Linear(64*25*25, 1*28*28)
        self.conv1 = nn.ConvTranspose2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.activation = nn.LeakyReLU(0.1)
        self.end_activation = end_activation
    
    def forward(self, x):
        n = x.shape[0]
        h1 = self.activation(self.fc1(x))
        h2 = self.drop(self.activation(self.conv1(h1.view(-1, 1, 7, 7))))
        h3 = self.drop(self.activation(self.conv2(h2)))
        h4 = self.drop(self.activation(self.conv3(h3)))
        h5 = self.end_activation(self.fc2(h4.view(n, -1)))
        return h5.view(-1, *self.output_shape)
        
#%%
if __name__ == '__main__':
    pass