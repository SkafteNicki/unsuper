#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 12:44:00 2018

@author: nsde
"""

#%%
import torch
from torch import nn
import datetime, os

#%%
class VAEGAN:
    def __init__(self, input_shape, latent_dim, encoder, decoder, discriminator, 
                 device='cpu', logdir=None):
        # Initialize
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.logdir = './logs/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') \
                        if logdir is None else logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        
        # Initialize encoder, decoder and discriminator and make full vae model
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        assert isinstance(self.encoder, nn.Module), 'Encoder is not a nn.Module'
        assert isinstance(self.decoder, nn.Module), 'Decoder is not a nn.Module'
        assert isinstance(self.discriminator, nn.Module), 'Discriminator is not a nn.Module'
        
    
    def train(self, trainloader, n_epochs, learning_rate=1e-3):
        # Make optimizers
        optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        optim_decoder = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate)
        optim_discrim = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        
        for epoch in range(n_epochs):