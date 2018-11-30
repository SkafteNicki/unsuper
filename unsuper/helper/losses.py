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
c = - 0.5 * math.log(2*math.pi)

#%%
def vae_loss(x, x_mu, x_var, z, z_mus, z_vars, eq_samples, iw_samples, 
             latent_dim, epoch, warmup, outputdensity):
    """ Calculates the ELBO for a variational autoencoder
    Arguments:
        x: input data [batch_size, *input_dim]
        x_mu: mean reconstruction [batch_size x eq_samples x iw_samples, *input_dim]
        x_var: variance of reconstruction [batch_size x eq_samples x iw_samples, *input_dim]
        z: latent variable 
        mus: mean in latent space  
        logvars: log variance in latent space
        eq_samples: int, number of equality samples
        iw_samples: int, number of importance weighted samples
        latent_dim: int, size of the latent space
        epoch: int, which epoch we are at
        warmup: int, how many warmup epoch to do
        outputdensity: str, output density of generative model
    Output:
        lower_bound: lower bound that should be maximized
        recon_term: reconstruction term for the ELBO
        kl_term: kl terms (multiple if multiple latents) in the ELBO term
    """
    eps = 1e-5 # to control underflow in variance estimates
    weight =  kl_scaling(epoch, warmup)
    
    batch_size = x.shape[0]
    x = x.view(batch_size, 1, 1, -1)
    x_mu = x_mu.view(batch_size, eq_samples, iw_samples, -1)
    x_var = x_var.view(batch_size, eq_samples, iw_samples, -1)

    z = [zs.view(-1, eq_samples, iw_samples, latent_dim) for zs in z]
    z_mus = [z_mus[0].view(-1, 1, 1, latent_dim)] + [m.view(-1, eq_samples, iw_samples, latent_dim) for m in z_mus[1:]]
    z_vars = [z_vars[0].view(-1, 1, 1, latent_dim)] + [l.view(-1, eq_samples, iw_samples, latent_dim) for l in z_vars[1:]]
    
    log_pz = [log_stdnormal(zs) for zs in z]
    log_qz = [log_normal2(zs, m, torch.log(l+eps)) for zs,m,l in zip(z, z_mus, z_vars)]
    
    if outputdensity == 'bernoulli':
        x_mu = x_mu.clamp(1e-5, 1-1e-5)
        log_px = (x * x_mu.log() + (1-x) * (1-x_mu).log())
    elif outputdensity == 'gaussian':
        log_px = log_normal2(x, x_mu, torch.log(x_var+eps), eps)
    else:
        ValueError('Unknown output density')
    a = log_px.sum(dim=3) + weight*(sum([p.sum(dim=3) for p in log_pz]) - sum([p.sum(dim=3) for p in log_qz]))
    a_max = torch.max(a, dim=2, keepdim=True)[0] #(batch_size, nsamples, 1)
    lower_bound = torch.mean(a_max) + torch.mean(torch.log(torch.mean(torch.exp(a-a_max), dim=2)))
    recon_term = log_px.sum(dim=3).mean()
    kl_term = [(lp-lq).sum(dim=3).mean() for lp,lq in zip(log_pz, log_qz)]
    return lower_bound, recon_term, kl_term

#%%
def log_stdnormal(x):
    """ Log probability of standard normal distribution elementwise """
    return c - x**2 / 2
    
#%%
def log_normal2(x, mean, log_var, eps=0.0):
    """ Log probability of normal distribution elementwise """
    return c - log_var/2 - (x - mean)**2 / (2 * torch.exp(log_var) + eps)

#%%
def kl_scaling(epoch=None, warmup=None):
    """ Annealing term for the KL-divergence """
    if epoch is None or warmup is None:
        return 1
    else:
        return float(np.min([epoch / warmup, 1]))