# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:53:22 2018

@author: nsde
"""

#%%
import torch
from torch import nn
from torch.nn.functional import softplus
import numpy as np
from torchvision.utils import make_grid
from ..helper.utility import affine_decompose
from ..helper.spatial_transformer import get_transformer

#%%
class VITAE_CI(nn.Module):
    def __init__(self, input_shape, latent_dim, encoder, decoder, outputdensity, ST_type, **kwargs):
        super(VITAE_CI, self).__init__()
        # Constants
        self.input_shape = input_shape
        self.flat_dim = np.prod(input_shape)
        self.latent_dim = latent_dim
        self.latent_spaces = 2
        self.outputdensity = outputdensity
        
        # Spatial transformer
        self.stn = get_transformer(ST_type)(input_shape)
        self.ST_type = ST_type
        
        # Define encoder and decoder
        self.encoder1 = encoder(input_shape, latent_dim)
        self.z_mean1 = nn.Linear(self.encoder1.encoder_dim, self.latent_dim)
        self.z_var1 = nn.Linear(self.encoder1.encoder_dim, self.latent_dim)
        self.decoder1 = decoder(input_shape, latent_dim)
        self.theta_mean = nn.Linear(self.decoder1.decoder_dim, self.stn.dim())
        self.theta_var = nn.Linear(self.decoder1.decoder_dim, self.stn.dim())
        
        self.encoder2 = encoder(input_shape, latent_dim)
        self.z_mean2 = nn.Linear(self.encoder2.encoder_dim, self.latent_dim)
        self.z_var2 = nn.Linear(self.encoder2.encoder_dim, self.latent_dim)
        self.decoder2 = decoder(input_shape, latent_dim)
        self.x_mean = nn.Linear(self.decoder2.decoder_dim, self.flat_dim)
        self.x_var = nn.Linear(self.decoder2.decoder_dim, self.flat_dim)
        
        # Define outputdensities
        if outputdensity == 'bernoulli':
            self.outputnonlin = torch.sigmoid
        elif outputdensity == 'gaussian':
            self.outputnonlin = lambda x: x
        else:
            ValueError('Unknown output density')
            
    #%%
    def encode1(self, x):
        enc = self.encoder1(x)
        z_mu = self.z_mean1(enc)
        z_var = self.z_var1(enc)
        return z_mu, softplus(z_var)
    
    #%%
    def decode1(self, z):
        dec = self.decoder1(z)
        theta_mean = self.theta_mean(dec)
        theta_var = self.theta_var(dec)
        return theta_mean, softplus(theta_var)
    
    #%%
    def encode2(self, x):
        enc = self.encoder2(x)
        z_mu = self.z_mean2(enc)
        z_var = self.z_var2(enc)
        return z_mu, softplus(z_var)
    
    #%%
    def decode2(self, z):
        dec = self.decoder2(z)
        x_mean = self.x_mean(dec).view(-1, *self.input_shape)
        x_var = self.x_var(dec).view(-1, *self.input_shape)
        return self.outputnonlin(x_mean), softplus(x_var)

    #%%
    def reparameterize(self, mu, var, eq_samples=1, iw_samples=1):
        batch_size, latent_dim = mu.shape
        eps = torch.randn(batch_size, eq_samples, iw_samples, latent_dim, device=var.device)
        return (mu[:,None,None,:] + var[:,None,None,:].sqrt() * eps).reshape(-1, latent_dim)
    
    #%%
    def forward(self, x, eq_samples=1, iw_samples=1):
        # Encode/decode transformer space
        mu1, var1 = self.encode1(x)
        z1 = self.reparameterize(mu1, var1, eq_samples, iw_samples)
        theta_mean, theta_var = self.decode1(z1)
        
        # Transform input
        x_new = self.stn(x.repeat(eq_samples*iw_samples, 1, 1, 1), -theta_mean)
        
        # Encode/decode semantic space
        mu2, var2 = self.encode2(x_new)
        z2 = self.reparameterize(mu2, var2, 1, 1)
        x_mean, x_var = self.decode2(z2)
        
        # "Detransform" output
        x_mean = self.stn(x_mean, theta_mean)
        x_var = self.stn(x_var, theta_mean)
        
        return x_mean, x_var, [z1, z2], [mu1, mu2], [var1, var2]

    #%%
    def sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z1 = torch.randn(n, self.latent_dim, device=device)
            z2 = torch.randn(n, self.latent_dim, device=device)
            theta_mean, theta_var = self.decode1(z1)
            x_mean, x_var = self.decode2(z2)
            out_mean = self.stn(x_mean, theta_mean)
            return out_mean

    #%%
    def sample_only_trans(self, n, img):
        device = next(self.parameters()).device
        with torch.no_grad():
            img = img.repeat(n, 1, 1, 1).to(device)
            z1 = torch.randn(n, self.latent_dim, device=device)
            theta_mean, theta_var = self.decode1(z1)
            out = self.stn(img, theta_mean)
            return out

    #%%
    def sample_only_images(self, n, trans):
        device = next(self.parameters()).device
        with torch.no_grad():
            trans = trans[None, :].repeat(n, 1).to(device)
            z2 = torch.randn(n, self.latent_dim, device=device)
            x_mean, x_var = self.decode2(z2)
            out = self.stn(x_mean, trans)
            return out
    
    #%%
    def sample_transformation(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z1 = torch.randn(n, self.latent_dim, device=device)
            theta_mean, theta_var = self.decode1(z1)
            theta = self.stn.trans_theta(theta_mean.reshape(-1, 2, 3))
            return theta.reshape(-1, 6)
    
    #%%
    def latent_representation(self, x):
        mu1, var1 = self.encode1(x)
        mu2, var2 = self.encode2(x)
        z1 = self.reparameterize(mu1, var1)
        z2 = self.reparameterize(mu2, var2)
        return [z1, z2]

    #%%
    def callback(self, writer, loader, epoch):
        n = 10      
        trans = torch.tensor([0,0,0,0,0,0], dtype=torch.float32)
        samples = self.sample_only_images(n*n, trans)
        writer.add_image('samples/fixed_trans', make_grid(samples.cpu(), nrow=n),
                         global_step=epoch)
        
        img = next(iter(loader))[0][0]
        samples = self.sample_only_trans(n*n, img)
        writer.add_image('samples/fixed_img', make_grid(samples.cpu(), nrow=n),
                          global_step=epoch)
    
        # Lets log a histogram of the transformation
        theta = self.sample_transformation(1000)
        for i in range(6):
            writer.add_histogram('transformation/a' + str(i), theta[:,i], 
                                 global_step=epoch, bins='auto')
            writer.add_scalar('transformation/mean_a' + str(i), theta[:,i].mean(),
                              global_step=epoch)
        
        # Also to a decomposition of the matrix and log these values
        values = affine_decompose(theta.view(-1, 2, 3))
        tags = ['sx', 'sy', 'm', 'theta', 'tx', 'ty']
        for i in range(6):
            writer.add_histogram('transformation/' + tags[i], values[i],
                                 global_step=epoch, bins='auto')
            writer.add_scalar('transformation/mean_' + tags[i], values[i].mean(),
                              global_step=epoch)
            
        # If 2d latent space we can make a fine meshgrid of sampled points
        if self.latent_dim == 2:
            device = next(self.parameters()).device
            x = np.linspace(-3, 3, 20)
            y = np.linspace(-3, 3, 20)
            z = np.stack([array.flatten() for array in np.meshgrid(x,y)], axis=1)
            z = torch.tensor(z, dtype=torch.float32)
            trans = torch.tensor([0,0,0,0,0,0], dtype=torch.float32).repeat(20*20, 1, 1)
            x_mean, x_var = self.decode2(z.to(device))
            out = self.stn(x_mean, trans.to(device))
            writer.add_image('samples/meshgrid', make_grid(out.cpu(), nrow=20),
                             global_step=epoch)
