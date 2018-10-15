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

from unsuper.helper.trainer import vae_trainer
from unsuper.data.mnist_data_loader import mnist_data_loader
from unsuper.models import VAE, VITAE, VITAE2
from unsuper.helper.encoders_and_decoders import MLP_Encoder, MLP_Decoder
from unsuper.helper.spatial_transformer import STN_AffineDiff
from unsuper.helper.callback import callback_vitae

#%%
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vitae', help='model to train')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
    parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
    parser.add_argument('--latent_dim', type=int, default=32, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--warmup', type=int, default=50, help='number of warmup epochs for kl-terms')
    parser.add_argument('--classes','--list', nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='classes to train on')
    args = parser.parse_args()
    return args

#%%
if __name__ == '__main__':
    # Input arguments
    args = argparser()
    img_size = (args.channels, args.img_size, args.img_size)
    
    # Logdir for results
    logdir = 'res/' + args.model + '/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    
    # Load data
    trainloader, testloader = mnist_data_loader(root='unsuper/data', 
                                                transform=transforms.ToTensor(),
                                                download=True,
                                                classes=args.classes,
                                                batch_size=args.batch_size)
    
    # Construct model
    if args.model == 'vae':
        encoder = MLP_Encoder(input_shape=img_size, latent_dim=args.latent_dim, h_size=[256, 128, 64])
        decoder = MLP_Decoder(output_shape=img_size, latent_dim=args.latent_dim, h_size=[64, 128, 256])
        model = VAE(encoder, decoder)
        callback = None
    elif args.model == 'vitae' or args.model == 'vitae2':
        encoder1 = MLP_Encoder(input_shape=img_size, latent_dim=args.latent_dim, h_size=[256, 128, 64])
        decoder1 = MLP_Decoder(output_shape=img_size, latent_dim=args.latent_dim, h_size=[64, 128, 256])
        encoder2 = MLP_Encoder(input_shape=img_size, latent_dim=args.latent_dim, h_size=[256, 128, 64])
        decoder2 = MLP_Decoder(output_shape=(6, ), latent_dim=args.latent_dim, h_size=[64, 128, 256], end_activation=torch.nn.LeakyReLU(0.1))
        stn = STN_AffineDiff(input_shape=img_size)        
        if args.model == 'vitae':
            model = VITAE(encoder1, encoder2, decoder1, decoder2, stn)
        else:
            model = VITAE2(encoder1, encoder2, decoder1, decoder2, stn)
        callback = callback_vitae
    else:
        ValueError('unknown model')
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    Trainer = vae_trainer(img_size, model, optimizer)
    Trainer.fit(trainloader=trainloader, 
                n_epochs=args.n_epochs, 
                warmup=args.warmup, 
                logdir=logdir,
                testloader=testloader,
                callback=callback)