#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:56:53 2018

@author: nsde
"""

#%%
import tensorflow as tf
import numpy as np

#%%
class batchifier():
    ''' Small iterator that will cut the input data into smaller batches. Can
        then be used in a for-loop like:
            for x_batch in batchifier(X, 100):
                # x_batch.shape[0] = 100            
    '''
                
    def __init__(self, *arrays, batch_size, shuffel=True):
        assert all([a.shape[0] for a in arrays]), '''all input arrays must have
            equal length for the first dimension '''
        self.arrays = arrays
        self.batch_size = batch_size    
        self.counter = 0
        self.N = self.arrays[0].shape[0]
        if shuffel:
            idx = np.random.permutation(self.N)
            self.arrays = [array[idx] for array in self.arrays]
    
    def __iter__(self):
        while self.counter < self.N:
            yield [array[self.counter:self.counter+self.batch_size] for array
                   in self.arrays]
            self.counter += self.batch_size

    def __len__(self):
        return int(np.ceil(self.N / self.batch_size))
    
#%%
if __name__ == '__main__':
    X = np.random.randn(100, 28, 28)
    y = np.random.randn(100)
    
    for i, (xx, yy) in enumerate(batchifier(X, y, batch_size=3)):
        print(i, xx.shape, yy.shape)