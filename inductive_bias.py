# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 07:30:05 2019

@author: nsde
"""

#%%
import torch
import argparse, datetime
from torchvision import transforms
import numpy as np

from unsuper.trainer import vae_trainer
from unsuper.data.mnist_data_loader import mnist_data_loader
from unsuper.data.perception_data_loader import perception_data_loader
from unsuper.helper.utility import model_summary
from unsuper.helper.encoder_decoder import get_encoder, get_decoder
from unsuper.models import get_model

#%%
def argparser():
    """ Argument parser for the main script """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model settings
    ms = parser.add_argument_group('Model settings')
    ms.add_argument('--model', type=str, default='vae', help='model to train')
    ms.add_argument('--ed_type', type=str, default='mlp', help='encoder/decoder type')
    ms.add_argument('--stn_type', type=str, default='affinediff', help='transformation type to use')
    
    # Training settings
    ts = parser.add_argument_group('Training settings')
    ts.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
    ts.add_argument('--eval_epoch', type=int, default=1000, help='when to evaluate log(p(x))')
    ts.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    ts.add_argument('--warmup', type=int, default=1, help='number of warmup epochs for kl-terms')
    ts.add_argument('--lr', type=float, default=1e-4, help='learning rate for adam optimizer')
    
    # Hyper settings
    hp = parser.add_argument_group('Variational settings')
    hp.add_argument('--latent_dim', type=int, default=2, help='dimensionality of the latent space')
    hp.add_argument('--density', type=str, default='bernoulli', help='output density')    
    hp.add_argument('--eq_samples', type=int, default=1, help='number of MC samples over the expectation over E_q(z|x)')
    hp.add_argument('--iw_samples', type=int, default=1, help='number of importance weighted samples')
    
    # Dataset settings
    ds = parser.add_argument_group('Dataset settings')
    ds.add_argument('--classes','--list', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='classes to train on')
    ds.add_argument('--num_points', type=int, default=10000, help='number of points in each class')
    ds.add_argument('--logdir', type=str, default='res', help='where to store results')
    ds.add_argument('--dataset', type=str, default='mnist', help='dataset to use')
    
    # Parse and return
    args = parser.parse_args()
    return args

#%%
if __name__ == '__main__':
    # Input arguments
    args = argparser()
    
    # Logdir for results
    if args.logdir == '':
        logdir = 'res/' + args.model + '/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    else:
        logdir = 'res/' + args.model + '/' + args.logdir
    
    # Load data
    print('Loading data')
    transformations = transforms.Compose([ 
        transforms.RandomAffine(degrees=20, translate=(0.1,0.1)), 
        transforms.ToTensor(), 
    ])
    trainloader, testloader = mnist_data_loader(root='unsuper/data', 
                                                transform=transformations,
                                                download=True,
                                                classes=args.classes,
                                                num_points=args.num_points,
                                                batch_size=args.batch_size)
    img_size = (1, 28, 28)

    # Construct model
    model_class = get_model('vitae_ci')
    model = model_class(input_shape = img_size,
                        latent_dim = args.latent_dim, 
                        encoder = get_encoder(args.ed_type), 
                        decoder = get_decoder(args.ed_type), 
                        outputdensity = args.density,
                        ST_type = args.stn_type)
    
    # Summary of model
    #model_summary(model)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    Trainer = vae_trainer(img_size, model, optimizer)
    Trainer.fit(trainloader=trainloader, 
                n_epochs=args.n_epochs, 
                warmup=args.warmup, 
                logdir=logdir,
                testloader=testloader,
                eq_samples=args.eq_samples, 
                iw_samples=args.iw_samples, 
                eval_epoch=args.eval_epoch)
    
    # Save model
    torch.save(model.state_dict(), logdir + '/trained_model.pt')
    
    #%% build new dataset
    class sample_data(torch.utils.data.Dataset):
        def __init__(self, n):
            X=np.zeros((n, 1, 28, 28))
            y=np.zeros((n, ))
            self.latent = [np.zeros((n, 2)) for _ in range(model.latent_spaces)]
            for b in range(int(n/100)):
                x, zs = model.special_sample(100)
                X[100*b:100*(b+1)] = x
                for i in range(model.latent_spaces):
                    self.latent[i][100*b:100*(b+1)] = zs[i]
            self.data = torch.tensor(X)
            self.targets = torch.tensor(y)
            
        def __getitem__(self, index):
            img, target = self.data[index], int(self.targets[index])

            if self.transform is not None:
                img = self.transform(img)
    
            if self.target_transform is not None:
                target = self.target_transform(target)
    
            return img, target
                
    train = sample_data(10000)
    test = sample_data(1000)
    trainloader = torch.utils.data.DataLoader(train, args.batch_size, )
    testloader = torch.utils.data.DataLoader(test, args.batch_size, )
    
    #%% train second model
    # Logdir for results
    if args.logdir == '':
        logdir = 'res/' + args.model + '/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    else:
        logdir = 'res/' + args.model + '/' + args.logdir

    # Construct model
    model_class2 = get_model(args.model)
    model2 = model_class(input_shape = img_size,
                        latent_dim = args.latent_dim, 
                        encoder = get_encoder(args.ed_type), 
                        decoder = get_decoder(args.ed_type), 
                        outputdensity = args.density,
                        ST_type = args.stn_type)
    
    # Summary of model
    #model_summary(model)
    
    # Optimizer
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr)
    
    # Train model
    Trainer2 = vae_trainer(img_size, model2, optimizer2)
    Trainer2.fit(trainloader=trainloader, 
                 n_epochs=args.n_epochs, 
                 warmup=args.warmup, 
                 logdir=logdir,
                 testloader=testloader,
                 eq_samples=args.eq_samples, 
                 iw_samples=args.iw_samples, 
                 eval_epoch=args.eval_epoch)
    
    # Save model
    torch.save(model2.state_dict(), logdir + '/trained_model2.pt')
    
    #%% save latent codes
    latent1 = [np.zeros((10000, 2)) for _ in range(model2.latent_spaces)]
    latent2 = [np.zeros((1000, 2)) for _ in range(model2.latent_spaces)]
    for i in range(100):
        _, _, _, zs_tr, _ = model2.semantics(trainloader.dataset.data[100*i:100*(i+1)])
        zs_te = model2.semantics(testloader.dataset.data[10*i:10*(i+1)])
        for j in range(model2.latent_spaces):
            latent1[j][100*i:100*(i+1)] = zs_tr[j]
            latent2[j][10*i:10*(j+1)] = zs_te[j]
            
    np.save(logdir + '/latent1', [trainloader.latent, testloader.latent])
    np.save(logdir + '/latent2', [latent1, latent2])
    
    
    