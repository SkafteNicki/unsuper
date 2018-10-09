# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 07:14:40 2018

@author: nsde
"""

#%%
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch_expm import torch_expm3x3

#%%
class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, intermidian_size=400):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.flat_dim = np.prod(input_shape)
        self.fc1 = nn.Linear(self.flat_dim, intermidian_size)
        self.fc2 = nn.Linear(intermidian_size, self.latent_dim)
        self.fc3 = nn.Linear(intermidian_size, self.latent_dim)
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = x.view(-1, self.flat_dim)
        h = self.activation(self.fc1(x))
        mu = self.fc2(h)
        logvar = self.fc3(h)
        return mu, logvar
   
#%%     
class Decoder(nn.Module):
    def __init__(self, output_shape, latent_dim, intermidian_size=400):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.flat_dim = np.prod(output_shape)
        self.fc1 = nn.Linear(self.latent_dim, intermidian_size)
        self.fc2 = nn.Linear(intermidian_size, self.flat_dim)
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        h  = self.activation(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out.view(-1, *self.output_shape)
    
#%%
class STN(nn.Module):
    def __init__(self, input_shape):
        super(STN, self).__init__()
        self.input_shape = input_shape
        
    def forward(self, x, theta):
        theta = theta.view(-1, 2, 3)
        theta = torch_expm3x3(theta)
        output_size = torch.Size([x.shape[0], *self.input_shape])
        grid = F.affine_grid(theta, output_size)
        x = F.grid_sample(x, grid)
        return x

#%%
class VAE_with_STN(nn.Module):
    def __init__(self, encoder1, encoder2, decoder1, decoder2, stn):
        super(VAE_with_STN, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.stn = stn
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu
        
    def forward(self, x):
        # Encode to transformer space
        mu1, logvar1 = self.encoder1(x)
        z1 = self.reparameterize(mu1, logvar1)
        
        # Decode transformation
        theta = self.decoder1(z1)
        
        # Call STN
        x_new = self.stn(x, theta)
        
        # Encode image
        mu2, logvar2 = self.encoder2(x_new)
        z2 = self.reparameterize(mu2, logvar2)
        
        # Decode image
        dec = self.decoder2(z2)
        
        # Use inverse transformation to "detransform image"
        recon = self.stn(dec, -theta)
        
        return recon, mu1, logvar1, mu2, logvar2
    
#%%
        
    
    
    