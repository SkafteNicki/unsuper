#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:01:09 2018

@author: nsde
"""

import torch

def _hermite(A, B, C, D, t):
    a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
    b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
    c = A * (-0.5) + C * 0.5
    d = B
    return a*t*t*t + b*t*t + c*t + d

def own_grid_sampler(data, grid, ):
    '''
        input: batch (N x C x IH x IW) or (N x C x ID x IH x IW)
        grid: (N x OH x OW x 2) or (N x OD x OH x OW x 3)
              assume grid is interval [-1, 1]
    '''
    ndim = grid.shape[-1]
    if ndim == 2:
        # Problem size
        N = data.shape[0]
        C = data.shape[1]
        IH = data.shape[2]
        IW = data.shape[3]
        OH = grid.shape[1]
        OW = grid.shape[2]
        
        # Scale from [-1, 1] -> [0, 1] -> [0, width/height]
        x = (grid[:,:,:,0] + 1.0) / 2.0 * IW
        y = (grid[:,:,:,1] + 1.0) / 2.0 * IH
        
        # Do sampling
        x1 = x.floor().type(torch.int32)
        x0 = x1 - 1
        x2 = x1 + 1
        x3 = x1 + 2
        
        y2 = x.floor().type(torch.int32)
        y0 = y1 - 1
        y2 = y1 + 1
        y3 = y1 + 2
        
        # Clip by value
        x0.clamp_(0, IW - 1)
        x1.clamp_(0, IW - 1)
        x2.clamp_(0, IW - 1)
        x3.clamp_(0, IW - 1)
        
        y0.clamp_(0, IH - 1)
        y1.clamp_(0, IH - 1)
        y2.clamp_(0, IH - 1)
        y3.clamp_(0, IH - 1)
        
        # Take care of batch effect, such that we can do linear indexing
        base = (torch.arange(N)*IH*IW).repeat(OH*OW)
        base_y0 = base + y0 * IW
        base_y1 = base + y1 * IW
        base_y2 = base + y2 * IW
        base_y3 = base + y3 * IW
        
        idx00 = base_y0 + x0
        idx01 = base_y1 + x0
        idx02 = base_y2 + x0
        idx03 = base_y3 + x0
        
        idx10 = base_y0 + x1
        idx11 = base_y1 + x1
        idx12 = base_y2 + x1
        idx13 = base_y3 + x1
        
        idx20 = base_y0 + x2
        idx21 = base_y1 + x2
        idx22 = base_y2 + x2
        idx23 = base_y3 + x2
        
        idx30 = base_y0 + x3
        idx31 = base_y1 + x3
        idx32 = base_y2 + x3
        idx33 = base_y3 + x3
        
        # Gather data
        data_flat = data.view(-1, C)
        p00 = data_flat[idx00]
        p10 = data_flat[idx10]
        p20 = data_flat[idx20]
        p30 = data_flat[idx30]
        
        p01 = data_flat[idx10]
        p11 = data_flat[idx11]
        p21 = data_flat[idx21]
        p31 = data_flat[idx31]
        
        p02 = data_flat[idx02]
        p12 = data_flat[idx12]
        p22 = data_flat[idx22]
        p32 = data_flat[idx32]
        
        p03 = data_flat[idx03]
        p13 = data_flat[idx13]
        p23 = data_flat[idx23]
        p33 = data_flat[idx33]
        
        # Do interpolation
        
        
    elif ndim == 3:
        NotImplemented('Will get support in the future')
    else:
        NotImplemented('Unsupported dimension')

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
        logvar = F.softplus(self.fc3(h))
        return mu, logvar
        
#%%     
class Decoder(nn.Module):
    def __init__(self, output_shape, latent_dim, intermidian_size=400):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.flat_dim = np.prod(output_shape)
        self.activation = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(self.latent_dim, intermidian_size)
        self.fc2 = nn.Linear(intermidian_size, self.flat_dim)
        
    def forward(self, x):
        h  = self.activation(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out.view(-1, *self.output_shape)

#%%
class NewDecoder(nn.Module):
    def __init__(self, output_shape, latent_dim, intermidian_size=400):
        super(NewDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.flat_dim = np.prod(output_shape)
        self.activation = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(self.latent_dim, intermidian_size)
        self.fc2 = nn.Linear(intermidian_size, self.flat_dim)
        self.fc2.weight = torch.nn.Parameter(1e-6*torch.randn_like(self.fc2.weight))
        self.fc2.bias = torch.nn.Parameter(1e-6*torch.randn_like(self.fc2.bias))
        
    def forward(self, x):
        h = self.activation(self.fc1(x))
        out = self.activation(self.fc2(h))
        return out.view(-1, *self.output_shape)
    