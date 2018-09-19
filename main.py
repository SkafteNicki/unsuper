# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 06:55:24 2018

@author: nsde
"""

#%%
from unsuper.data.data_loader import mnist_train_loader, mnist_test_loader


#%%
if __name__ == '__main__':
    train_loader = mnist_train_loader(batch_size=10, transform=None)
    test_loader = mnist_test_loader(batch_size=10, transform=None)
