# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 06:56:08 2018

@author: nsde
"""

#%%
import torch
from torch import nn
from torch.autograd import Variable

#%%
class GAN(object):
    def __init__(self, latent_dim, Generator, Discriminator, device='cpu'):
        # Device to run on
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        
        # Initialize generator and discriminator
        self.generator = Generator()
        self.discriminator = Discriminator()
        assert isinstance(Generator, nn.Module), 'Generator is not a nn.Module'
        assert isinstance(Discriminator, nn.Module), 'Discriminator is not a nn.Module'
        
        # Loss function
        self.loss = torch.nn.BCELoss()
        
        # Transfer to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.loss.to(self.device)
        
    def train(self, dataloader, n_epochs, learning_rate):
        # Make optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        optimizer_D = torch.optim.Adam(self.discriminator.meters(), lr=learning_rate)
        
        for epoch in range(n_epochs):
            for i, (imgs, _) in enumerate(dataloader):
                # Adversarial ground truths
                valid = Variable(torch.Tensor(imgs.size(0), 1).fill_(1.0).to(self.device), requires_grad=False)
                fake = Variable(torch.Tensor(imgs.size(0), 1).fill_(0.0).to(self.device), requires_grad=False)
        
                # Configure input
                real_imgs = Variable(imgs.to(self.device))
                
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()
            
                # Sample noise as generator input
                z = Variable(torch.randn(imgs.shape[0], self.latent_dim).to(self.device))
                
                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.loss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                optimizer_G.step()
                
                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.loss(self.discriminator(real_imgs), valid)
                fake_loss = self.loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()
                
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

#%%
if __name__ == '__main__':
                    