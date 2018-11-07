#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:44:23 2018

@author: nsde
"""

#%%
from torch import nn
import numpy as np
from ..helper.utility import CenterCrop

#%%
def get_encoder(encoder_name):
    models = {'mlp': mlp_encoder,
              'conv': conv_encoder,
              }
    assert (encoder_name in models), 'Encoder not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
    return models[encoder_name]

#%%
def get_decoder(decoder_name):
    models = {'mlp': mlp_decoder,
              'conv': conv_decoder,
              }
    assert (decoder_name in models), 'Decoder not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
    return models[decoder_name]

#%%
class mlp_encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(mlp_encoder, self).__init__()
        self.flat_dim = np.prod(input_shape)
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
        self.encoder_dim = 256
        
    def forward(self, x):
        return self.encoder(x.view(x.shape[0], -1))

#%%
class mlp_decoder(nn.Module):
    def __init__(self, output_shape, latent_dim):
        super(mlp_decoder, self).__init__()
        self.flat_dim = np.prod(output_shape)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
        )
        self.decoder_dim = 512
        
    def forward(self, z):
        return self.decoder(z)

#%%
class conv_encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(conv_encoder, self).__init__()
        c,h,w = input_shape
        self.z_dim = int(np.ceil(h/2**2)) # receptive field downsampled 2 times
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.encoder_dim = (64, self.z_dim, self.z_dim)
    
    def forward(self, x):
        return self.encoder(x)
    
#%%
class conv_decoder(nn.Module):
    def __init__(self, output_shape, latent_dim):
        super(conv_decoder, self).__init__() 
        c,h,w = output_shape
        self.z_dim = int(np.ceil(h/2**2))        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * self.z_dim**2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1),
            CenterCrop(h,w),
        )
        self.decoder_dim = (1, h, w)
    
    def forward(self, z):
        return self.decoder(z)