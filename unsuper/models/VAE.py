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
    def __init__(self, input_shape, latent_dim, Encoder, Decoder, device='cpu', logdir=None):
        # Initialize
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.logdir = './logs/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') \
                        if logdir is None else logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        
        # Initialize encoder and decoder and make full vae model 
        self.encoder = Encoder()
        self.decoder = Decoder()
        assert isinstance(self.encoder, nn.Module), 'Encoder is not a nn.Module'
        assert isinstance(self.decoder, nn.Module), 'Decoder is not a nn.Module'
        self.combined = VAE_module(self.encoder, self.decoder)
        
        # Loss function
        self.loss_f = self._loss_function
        
        # Transfer to device
        self.combined.to(device)
    
    def train(self, trainloader, n_epochs, learning_rate=1e-3, testloader=None):
        # Make optimizers
        optimizer = torch.optim.Adam(self.combined.parameters(), lr=learning_rate)
        
        self.combined.train() # training mode
        train_loss = 0
        for epoch in range(n_epochs):
            progress_bar = tqdm(desc='Epoch ' + str(epoch), total=len(trainloader.dataset), 
                                unit='samples', ncols=70)
            for i, (data, _) in enumerate(trainloader):
                data = data.to(self.device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.combined(data)
                loss = self.loss_f(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                progress_bar.update(data.size(0))
                progress_bar.set_postfix({'loss': loss.item()})
            progress_bar.set_postfix({'Average loss': train_loss / len(trainloader)})
            progress_bar.close()
            
            # Save some res
            self.snapshot(postfix=epoch)
            self.reconstruct(data, postfix=epoch)
            
            if testloader:
                test_loss = 0
                self.combined.eval() # evaluation mode
                for i, (data, _) in enumerate(testloader):
                    data = data.to(self.device)
                    recon_batch, mu, logvar = self.combined(data)
                    test_loss += self.loss_f(recon_batch, data, mu, logvar).item()
                test_loss /= len(testloader)
                print('Test set loss: {:.4f}'.format(test_loss))
    
    def sample(self, n_samples):
        with torch.no_grad():
            sample = torch.randn(n_samples, self.latent_dim).to(self.device)
            sample = self.decoder(sample).cpu()
        return sample
    
    def snapshot(self, n=64, postfix=''):
        samples = self.sample(n)
        save_image(samples.view(n,*self.input_shape).cpu(), 
                   self.logdir + '/samples' + str(postfix) + '.png')
    
    def reconstruct(self, data, postfix=''):
        self.combined.eval() # evaluation mode
        recon_batch, _, _ = self.combined(data)
        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n], recon_batch.view(data.size(0), *self.input_shape)[:n]])
        save_image(comparison.cpu(),
                   self.logdir + '/reconstruction' + str(postfix) + '.png', nrow=n)
    
    def _loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, np.prod(self.input_shape)))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= recon_x.numel() # normalize KL divergence
        return BCE + KLD