#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:48:18 2018

@author: nsde
"""

#%%
from .vae_conv import VAE_Conv
from .vae_mlp import VAE_Mlp
from .vitae_conv import VITAE_Conv
from .vitae_mlp import VITAE_Mlp
#from .vitae2_conv import VITAE2_Conv
#from .vitae2_mlp import VITAE2_Mlp

#%%
def get_model(model_name):
    models = {'vae_conv': VAE_Conv,
              'vae_mlp': VAE_Mlp,
              'vitae_conv': VITAE_Conv,
              'vitae_mlp': VITAE_Mlp,
              #'vitae2_conv': VITAE2_Conv,
              #'vitae2_mlp': VITAE2_Mlp
              }
    assert (model_name in models), 'Transformer not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
    return models[model_name]