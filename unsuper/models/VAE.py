#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 07:57:55 2018

@author: nsde
"""

#%%
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import os
import datetime

#%%
class VAE_module(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE_module, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

#%%
class VAE:
    def __init__(self, input_shape, latent_dim, encoder, decoder, device='cpu', logdir=None):
        # Initialize
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.logdir = './logs/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') \
                        if logdir is None else logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        
        # Initialize encoder and decoder and make full vae model 
        self.encoder = encoder
        self.decoder = decoder 
        assert isinstance(self.encoder, nn.Module), 'Encoder is not a nn.Module'
        assert isinstance(self.decoder, nn.Module), 'Decoder is not a nn.Module'
        self.combined = VAE_module(self.encoder, self.decoder)
        
        # Loss function
        self.loss_f = self._loss_function
        
        # Transfer to device
        self.combined.to(device)
    
    #%%
    def train(self, trainloader, n_epochs, learning_rate=1e-3, testloader=None):
        # Make optimizers
        optimizer = torch.optim.Adam(self.combined.parameters(), lr=learning_rate)
        
        # Main loop 
        self.combined.train() # training mode 
        for epoch in range(n_epochs):
            progress_bar = tqdm(desc='Epoch ' + str(epoch), total=len(trainloader.dataset), 
                                unit='samples')
            train_loss = 0    
            # Training loop
            for i, (data, _) in enumerate(trainloader):
                optimizer.zero_grad()
                data = data.to(self.device)
                recon_batch, mu, logvar = self.combined(data)
                loss = self.loss_f(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                progress_bar.update(data.size(0))
                progress_bar.set_postfix({'loss': loss.item()})
                
            progress_bar.set_postfix({'Average loss': train_loss / len(trainloader)})
            progress_bar.close()
            
            if testloader:
                test_loss = 0
                self.combined.eval() # evaluation mode
                for i, (data, _) in enumerate(testloader):
                    data = data.to(self.device)
                    recon_batch, mu, logvar = self.combined(data)
                    test_loss += self.loss_f(recon_batch, data, mu, logvar).item()
                test_loss /= len(testloader)
                print('Test set loss: {:.4f}'.format(test_loss))
            
            # Save some res
            self.snapshot(data, postfix=epoch)
            
    #%%
    def sample(self, n_samples):
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            sample = self.decoder(z)
            sample = sample.view(-1, *self.input_shape)
        return sample

    #%%
    def reconstruct(self, data):
        with torch.no_grad():
            recon_data, _, _ = self.combined(data.to(self.device))
            recon_data = recon_data.view(-1, *self.input_shape)
        return recon_data
    
    #%%
    def snapshot(self, data, n=64, postfix=''):
        # Sample from latent space, and save generated images
        samples = self.sample(n)
        save_image(samples.cpu(), self.logdir + '/samples' + str(postfix) + '.png')
    
        # Reconstruct data, and save together with original images (max 10)
        n = min(data.size(0), 10)
        recon_data = self.reconstruct(data)
        comparison = torch.cat([data[:n], recon_data[:n]])
        save_image(comparison.cpu(), self.logdir + '/reconstruction' + str(postfix) + '.png', nrow=n)
    
    #%%
    def _loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, np.prod(self.input_shape)))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= recon_x.numel() # normalize KL divergence
        return BCE + KLD