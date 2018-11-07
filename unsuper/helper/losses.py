#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:39:59 2018

@author: nsde
"""

#%%
import numpy as np
import torch
import math

#%%
def vae_loss(x, x_mu, x_var, z, mus, logvars, eq_samples, iw_samples, 
             latent_dim, epoch, warmup, outputdensity):
    """ Calculates the ELBO for a variational autoencoder
    Arguments:
        x:
        x_mu:
        x_var:
        z:
        mus:
        logvars:
        eq_samples:
        iw_samples:
        latent_dim:
        epoch:
        warmup:
        outputdensity:
    Output:
        lower_bound:
        recon_term:
        kl_term:
    """    
    weight =  kl_scaling(epoch, warmup)
    
    batch_size = x.shape[0]
    x = x.view(batch_size, 1, 1, -1)
    x_mu = x_mu.view(batch_size, eq_samples, iw_samples, -1)
    x_var = x_var.view(batch_size, eq_samples, iw_samples, -1)

    z = [zs.view(-1, eq_samples, iw_samples, latent_dim) for zs in z]
    mus = [mus[0].view(-1, 1, 1, latent_dim)] + [m.view(-1, eq_samples, iw_samples, latent_dim) for m in mus[1:]]
    logvars = [logvars[0].view(-1, 1, 1, latent_dim)] + [l.view(-1, eq_samples, iw_samples, latent_dim) for l in logvars[1:]]
    
    log_pz = [log_stdnormal(zs) for zs in z]
    log_qz = [log_normal2(zs, m, l) for zs,m,l in zip(z, mus, logvars)]
    
    if outputdensity == 'bernoulli':
        x_mu = x_mu.clamp(1e-5, 1-1e-5)
        log_px = (x * x_mu.log() + (1-x) * (1-x_mu).log())
    elif outputdensity == 'gaussian':
        log_px = log_normal2(x, x_mu, torch.log(x_var+1e-5))
    else:
        ValueError('Unknown output density')

    a = log_px.sum(dim=3) + weight*(sum([p.sum(dim=3) for p in log_pz]) - sum([p.sum(dim=3) for p in log_qz]))
    a_max = torch.max(a, dim=2, keepdim=True)[0] #(batch_size, nsamples, 1)
    lower_bound = torch.mean(a_max) + torch.max(torch.log(torch.mean(torch.exp(a-a_max), dim=2)))
    recon_term = log_px.sum(dim=3).mean()
    kl_term = [(lp-lq).sum(dim=3).mean() for lp,lq in zip(log_pz, log_qz)]
    return lower_bound, recon_term, kl_term

#%%
def log_stdnormal(x):
    """ Log probability of standard normal distribution elementwise """
    c = - 0.5 * math.log(2*math.pi)
    return c - x**2 / 2
    
#%%
def log_normal2(x, mean, log_var, eps=0.0):
    """ Log probability of normal distribution elementwise """
    c = - 0.5 * math.log(2*math.pi)
    return c - log_var/2 - (x - mean)**2 / (2 * torch.exp(log_var) + eps)

#%%
def kl_scaling(epoch=None, warmup=None):
    """ Annealing term for the KL-divergence """
    if epoch is None or warmup is None:
        return 1
    else:
        return float(np.min([epoch / warmup, 1]))