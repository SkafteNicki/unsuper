#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:39:59 2018

@author: nsde
"""

#%%
import numpy as np
import torch
import torch.nn.functional as F

#%%
def reconstruction_loss(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return BCE

#%%
def kullback_leibler_divergence(mus, logvars):
    KLD = [-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) for
           mu, logvar in zip(mus, logvars)]
    return KLD

#%%
def kl_scaling(epoch=None, warmup=None):
    if epoch is None or warmup is None:
        return 1
    else:
        return float(np.min([epoch / warmup, 1]))