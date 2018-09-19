#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:48:57 2018

@author: nsde
"""

#%%
import tensorflow as tf
from tensorflow import keras
import numpy as np
from ..helper.utility import batchifier

#%%
class GAN:
    def __init__(self, img_shape, latent_space = 100):
        self.img_shape = img_shape
        self.latent_space = latent_space
    
    #%%
    def compile(self, optimizer = 'adam', learning_rate = 1e-4):
        optimizer = tf.keras.optimizers.get(optimizer)
        self.optimizer = optimizer(lr = learning_rate)
        
        # Generator
        self.generator = self._build_generator()
        
        # Discriminator
        self.discriminator = self._build_discriminator()
        self.discriminator.build(loss='binary_crossentropy',
                                 optimizer=optimizer,
                                 metrics=['accuracy'])
        
        # Generator takes noise as input and generates images
        z = keras.layers.Input(shape=(self.latent_space,))
        img = self.generator(z)
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        
        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = keras.Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    #%%
    def train(self, X, y, epochs = 10, batch_size = 128, ):
        
        
        
        for e in range(epochs):
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            for i, (x_batch, y_batch) in batchifier(X, y, batch_size = batch_size):
                # Train discriminator
                noise = np.random.normal(0, 1, (batch_size, self.latent_space))
                gen_imgs = self.generator.predict(noise)
                d_loss_real = self.discriminator.train_on_batch(x_batch, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # Train generator
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                g_loss = self.combined.train_on_batch(noise, valid)
        
    
    #%%
    def _build_generator(self):
        
        model = keras.Sequential()

        model.add(keras.layers.Dense(256, input_dim=self.latent_dim))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Dense(1024))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(keras.layers.Reshape(self.img_shape))

        model.summary()

        noise = keras.layers.Input(shape=(self.latent_dim,))
        img = model(noise)

        return keras.Model(noise, img)
    
    #%%        
    def _build_discriminator(self):
        
        model = keras.Sequential()

        model.add(keras.layers.Flatten(input_shape=self.img_shape))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        model.summary()

        img = keras.layers.Input(shape=self.img_shape)
        validity = model(img)

        return keras.Model(img, validity)
        
    

#%%
if __name__ == '__main__':
    pass