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
def ELBO_adjusted(x, recon_x, mus, logvars, mcmc_samples, epoch=None, warmup=None):
    # Calculate terms
    weight = kl_scaling(epoch, warmup)
    NBCE = reconstruction_loss(recon_x, x)
    mu = [m.view(-1, x.shape[0], m.shape[1]) for m in mus]
    logvar = [l.view(-1, x.shape[0], l.shape[1]) for l in logvars]
    KLD = kullback_leibler_divergence(mu, logvar, dim=[1,2])
    KLD = [kl.mean() for kl in KLD]
    # Sum
    loss = NBCE + weight*sum(KLD)
    return loss, NBCE, KLD

#%%
def ELBO(x, recon_x, mus, logvars, epoch=None, warmup=None):
    # Calculate terms
    weight = kl_scaling(epoch, warmup)
    NBCE = reconstruction_loss(recon_x, x)
    KLD = kullback_leibler_divergence(mus, logvars)
    # Sum
    loss = NBCE + weight*sum(KLD)
    return loss, NBCE, KLD

#%%
def reconstruction_loss(recon_x, x):
    NBCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return NBCE

#%%
def kullback_leibler_divergence(mus, logvars, dim=[0,1]):
    KLD = [-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=dim) for
           mu, logvar in zip(mus, logvars)]
    return KLD

#%%
def kl_scaling(epoch=None, warmup=None):
    if epoch is None or warmup is None:
        return 1
    else:
        return float(np.min([epoch / warmup, 1]))