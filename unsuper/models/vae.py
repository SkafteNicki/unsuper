#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:33:13 2018

@author: nsde
"""

#%%
import torch
from torch import nn
from torch.nn.functional import softplus
import numpy as np
from torchvision.utils import make_grid

#%%
class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim, encoder, decoder, outputdensity, **kwargs):
        super(VAE, self).__init__()
        # Constants
        self.input_shape = input_shape
        self.flat_dim = np.prod(input_shape)
        self.latent_dim = latent_dim
        self.latent_spaces = 1
        self.outputdensity = outputdensity
        
        # Define encoder and decoder
        self.encoder = encoder(input_shape, latent_dim)
        self.z_mean = nn.Linear(self.encoder.encoder_dim, self.latent_dim)
        self.z_var = nn.Linear(self.encoder.encoder_dim, self.latent_dim)
        self.decoder = decoder(input_shape, latent_dim)
        self.x_mean = nn.Linear(self.decoder.decoder_dim, self.flat_dim)
        self.x_var = nn.Linear(self.decoder.decoder_dim, self.flat_dim)
        
        # Define outputdensities
        if outputdensity == 'bernoulli':
            self.outputnonlin = torch.sigmoid
        elif outputdensity == 'gaussian':
            self.outputnonlin = lambda x: x
        else:
            ValueError('Unknown output density')
    
    #%%
    def encode(self, x):
        enc = self.encoder(x)
        z_mu = self.z_mean(enc)
        z_var = self.z_var(enc)
        return z_mu, softplus(z_var)
    
    #%%
    def decode(self, z):
        dec = self.decoder(z)
        x_mu = self.x_mean(dec).view(-1, *self.input_shape)
        x_var = self.x_var(dec).view(-1, *self.input_shape)
        return self.outputnonlin(x_mu), softplus(x_var)
    
    #%%
    def reparameterize(self, mu, var, eq_samples=1, iw_samples=1):
        batch_size, latent_dim = mu.shape
        eps = torch.randn(batch_size, eq_samples, iw_samples, latent_dim, device=var.device)
        return (mu[:,None,None,:] + var[:,None,None,:].sqrt() * eps).reshape(-1, latent_dim)
    
    #%%
    def forward(self, x, eq_samples=1, iw_samples=1):
        z_mu, z_var = self.encode(x)
        z = self.reparameterize(z_mu, z_var, eq_samples, iw_samples)
        x_mu, x_var = self.decode(z)
        return x_mu, x_var, [z], [z_mu], [z_var]
    
    #%%
    def sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            x_mu, x_var = self.decode(z)
            return x_mu
    
    #%%
    def latent_representation(self, x):
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        return [z]
    
    #%%
    def callback(self, writer, loader, epoch):
        # If 2d latent space we can make a fine meshgrid of sampled points
        if self.latent_dim == 2:
            device = next(self.parameters()).device
            x = np.linspace(-3, 3, 20)
            y = np.linspace(-3, 3, 20)
            z = np.stack([array.flatten() for array in np.meshgrid(x,y)], axis=1)
            z = torch.tensor(z, dtype=torch.float32)
            out_mu, out_var = self.decode(z.to(device))
            writer.add_image('samples/meshgrid', make_grid(out_mu.cpu(), nrow=20),
                             global_step=epoch)
    
#%%
if __name__ == '__main__':
    pass