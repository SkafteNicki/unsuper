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

from ..helper.utility import log_normal2

from ..helper.losses import ELBO

#%%
class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim, encoder, decoder, outputdensity):
        super(VAE, self).__init__()
        # Constants
        self.input_shape = input_shape
        self.flat_dim = np.prod(input_shape)
        self.latent_dim = latent_dim
        
        # Define encoder and decoder
        self.encoder = encoder(latent_dim)
        self.z_mean = nn.Linear(self.encoder.encoder_dim, self.latent_dim)
        self.z_var = nn.Linear(self.encoder.encoder_dim, self.latent_dim)
        self.decoder = decoder(latent_dim)
        self.x_mean = nn.Linear(self.decoder.decoder_dim, self.flat_dim)
        self.x_var = nn.Linear(self.decoder.decoder_dim, self.flat_dim)
        
        # Define outputdensities
        if outputdensity == 'bernoulli':
            self.outputnonlin = torch.sigmoid
            self.recon_loss = torch.nn.functional.binary_cross_entropy
        if outputdensity == 'gaussian':
            self.outputnonlin = lambda x: x
            self.recon_loss = log_normal2
    
    #%%
    def encode(self, x):
        enc = self.encoder(x.view(x.shape[0], -1))
        z_mu = self.z_mean(enc)
        z_logvar = softplus(self.z_var(enc))
        return z_mu, softplus(z_logvar)
    
    #%%
    def decode(self, z):
        dec = self.decoder(z)
        x_mu = self.x_mean(dec).view(-1, *self.input_shape)
        x_var = self.x_var(dec).view(-1, *self.input_shape)
        return self.outputnonlin(x_mu), softplus(x_var)
    
    #%%
    def reparameterize(self, mu, logvar, iw_samples=1):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn(iw_samples, *std.shape, device=std.device)
            return eps.mul(std).add(mu).reshape(-1, std.shape[1])
        else:
            return mu.repeat(iw_samples, 1)
    
    #%%
    def forward(self, x, iw_samples=1):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, iw_samples)
        x_mu, x_var = self.decode(z)
        return x_mu, x_var, [mu], [logvar]
    
    #%%
    def sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            x_mu, x_var = self.decode(z)
            return x_mu, x_var
    
    #%%
    def latent_representation(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return [z]
    
    #%%
    def loss_f(self, data, recon_data, mus, logvars, epoch, warmup):
        return ELBO(data, recon_data, mus, logvars, epoch, warmup)
    
    #%%
    def __len__(self):
        return 1
    
    #%%
    def callback(self, writer, loader, epoch):
        # If 2d latent space we can make a fine meshgrid of sampled points
        if self.latent_dim == 2:
            device = next(self.parameters()).device
            x = np.linspace(-3, 3, 20)
            y = np.linspace(-3, 3, 20)
            z = np.stack([array.flatten() for array in np.meshgrid(x,y)], axis=1)
            z = torch.tensor(z, dtype=torch.float32)
            out = self.decode(z.to(device))
            writer.add_image('samples/meshgrid', make_grid(out.cpu(), nrow=20),
                             global_step=epoch)
    
#%%
if __name__ == '__main__':
    pass


