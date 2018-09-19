#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:56:53 2018

@author: nsde
"""
#%%
import os

#%%
def get_dir(file):
    """ Get the folder of specified file """
    return os.path.dirname(os.path.realpath(file))
    
#%%
if __name__ == '__main__':
    pass