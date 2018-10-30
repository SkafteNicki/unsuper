#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:18:28 2018

@author: nsde
"""

#%%
import torch
import argparse, datetime
from torchvision import transforms

from unsuper.trainer import vae_trainer
from unsuper.data.mnist_data_loader import mnist_data_loader
from unsuper.models import get_model
from unsuper.helper.utility import model_summary

#%%
def argparser():
    """ Argument parser for the main script """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vitae', help='model to train')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
    parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
    parser.add_argument('--latent_dim', type=int, default=32, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--warmup', type=int, default=1, help='number of warmup epochs for kl-terms')
    parser.add_argument('--classes','--list', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='classes to train on')
    parser.add_argument('--num_points', type=int, default=10000, help='number of points in each class')
    args = parser.parse_args()
    return args

#%%
if __name__ == '__main__':
    # Input arguments
    args = argparser()
    img_size = (args.channels, args.img_size, args.img_size)
    
    # Logdir for results
    logdir = 'res/' + args.model + '/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    
    transformations = transforms.Compose([ 
            #transforms.Pad(padding=7, fill=0),
            transforms.RandomAffine(degrees=20, translate=(0.1,0.1)), 
            transforms.ToTensor(), 
    ])
    # Load data
    trainloader, testloader = mnist_data_loader(root='unsuper/data', 
                                                transform=transformations,
                                                download=True,
                                                classes=args.classes,
                                                num_points=args.num_points,
                                                batch_size=args.batch_size)
    
    # Construct model
    model = get_model(args.model)(input_shape=img_size, latent_dim=args.latent_dim)
    
    # Summary of model
    model_summary(model)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    Trainer = vae_trainer(img_size, model, optimizer)
    Trainer.fit(trainloader=trainloader, 
                n_epochs=args.n_epochs, 
                warmup=args.warmup, 
                logdir=logdir,
                testloader=testloader)
    
    # Save model
    torch.save(model.state_dict(), logdir + '/trained_model.pt')
