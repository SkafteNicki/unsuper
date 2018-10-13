#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:02:37 2018

@author: nsde
"""
#%%
import torch
from torch import nn
from ..helper.expm import torch_expm

#%%
def _expm(theta):
    n_theta = theta.shape[0]
    zero_row = torch.zeros(n_theta, 1, 3, dtype=theta.dtype, device=theta.device)
    theta = torch.cat([theta, zero_row], dim=1)
    theta = torch_expm(theta)
    theta = theta[:,:2,:]
    return theta

#%%
class VITAE(nn.Module):
    def __init__(self, encoder1, encoder2, decoder1, decoder2, stn):
        super(VITAE, self).__init__()
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
        mu2, logvar2 = self.encoder2(x)
        z2 = self.reparameterize(mu2, logvar2)
        
        # Decode transformation
        theta = self.decoder2(z2)
        
        # Call STN with inverse transformation
        x_new = self.stn(x, -theta)
        
        # Encode image
        mu1, logvar1 = self.encoder1(x_new)
        z1 = self.reparameterize(mu1, logvar1)
        
        # Decode image
        dec = self.decoder1(z1)
        
        # Use inverse transformation to "detransform image"
        recon = self.stn(dec, theta)
        
        return recon, [mu1, mu2], [logvar1, logvar2]
        
    def sample_only_images(self, n, trans):
        device = next(self.parameters()).device
        with torch.no_grad():
            trans = trans[None, :].repeat(n, 1).to(device)
            z1 = torch.randn(n, self.encoder1.latent_dim, device=device)
            dec = self.decoder1(z1)
            out = self.stn(dec, trans)
            return out
        
    def sample_only_trans(self, n, img):
        device = next(self.parameters()).device
        with torch.no_grad():
            img = img.repeat(n, 1, 1, 1).to(device)
            z2 = torch.randn(n, self.encoder2.latent_dim, device=device)
            theta = self.decoder2(z2)
            out = self.stn(img, theta)
            return out
    
    def latent_representation(self, x):
        mu1, logvar1 = self.encoder1(x)
        mu2, logvar2 = self.encoder2(x)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        return z1, z2
    
    def sample_transformation(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z2 = torch.randn(n, self.encoder2.latent_dim, device=device)
            theta = self.decoder2(z2)
            theta = _expm(theta.reshape(-1, 2, 3))
            return theta.reshape(-1, 6)