#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:18:45 2018

@author: nsde
"""
#%%
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import time, os, datetime
from tensorboardX import SummaryWriter
from .losses import reconstruction_loss, kullback_leibler_divergence, kl_scaling

#%%
class vae_trainer:
    def __init__(self, input_shape, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.input_shape = input_shape
        
        # Get the device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        # Move model to gpu (if avaible)
        if torch.cuda.is_available():
            self.model.cuda()
    
    #%%
    def fit(self, trainloader, n_epochs=10, warmup=None, logdir='',
            testloader=None, callback=None):
        # Dir to log results
        logdir = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') if logdir is None else logdir
        if not os.path.exists(logdir): os.makedirs(logdir)
        
        # Summary writer
        writer = SummaryWriter(log_dir=logdir)
        
        start = time.time()
        # Main loop
        for epoch in range(1, n_epochs+1):
            progress_bar = tqdm(desc='Epoch ' + str(epoch), total=len(trainloader.dataset), 
                                unit='samples')
            train_loss = 0
            weight = kl_scaling(epoch, warmup)
            # Training loop
            self.model.train()
            for i, (data, _) in enumerate(trainloader):
                # Zero gradient
                self.optimizer.zero_grad()
            
                # Feed forward data
                data = data.reshape(-1, *self.input_shape).to(self.device)
                recon_data, mus, logvars = self.model(data)
                
                # Calculat loss
                recon_term = reconstruction_loss(recon_data, data)
                kl_terms = kullback_leibler_divergence(mus, logvars)
                loss = recon_term + weight*sum(kl_terms)
                train_loss += loss.item()
                
                # Backpropegate and optimize
                loss.backward()
                self.optimizer.step()
                
                # Write to consol
                progress_bar.update(data.size(0))
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Save to tensorboard
                iteration = epoch*len(trainloader) + i
                writer.add_scalar('train/total_loss', loss, iteration)
                writer.add_scalar('train/recon_loss', recon_term, iteration)
                for j, kl_loss in enumerate(kl_terms):
                    writer.add_scalar('train/KL_loss' + str(j), kl_loss, iteration)
            
            progress_bar.set_postfix({'Average loss': train_loss / len(trainloader)})
            progress_bar.close()
            writer.scalar_dict
            # Log for the training set
            n = 10
            data_train = next(iter(trainloader))[0].to(self.device)[:n]
            recon_data_train = self.model(data_train)[0]
            writer.add_image('train/recon', make_grid(torch.cat([data_train, 
                         recon_data_train]).cpu(), nrow=n), global_step=epoch)
            samples = self.model.sample(n*n)    
            writer.add_image('samples/samples', make_grid(samples.cpu(), nrow=n), 
                             global_step=epoch)
            
            if testloader:
                # Evaluate on test set
                self.model.eval()
                recon_term, kl_terms = 0, len(kl_terms)*[0]
                for i, (data, _) in enumerate(testloader):
                    data = data.reshape(-1, *self.input_shape).to(self.device)
                    recon_data, mus, logvars = self.model(data)    
                    recon_term += reconstruction_loss(recon_data, data)
                    kl_terms = [l1 + l2 for l1,l2 in zip(kullback_leibler_divergence(mus, logvars), kl_terms)]
                    
                test_loss = recon_term + weight*sum(kl_terms)
        
                writer.add_scalar('test/total_loss', test_loss, iteration)
                writer.add_scalar('test/recon_loss', recon_term, iteration)
                for j, kl_loss in enumerate(kl_terms):
                    writer.add_scalar('test/KL_loss' + str(j), kl_loss, iteration)
        
                data_test = next(iter(testloader))[0].to(self.device)[:n]
                recon_data_test = self.model(data_test)[0]
                writer.add_image('test/recon', make_grid(torch.cat([data_test, 
                         recon_data_test]).cpu(), nrow=n), global_step=epoch)
                
                # If callback call it now
                if callback:
                    callback(writer, self.model, epoch, data_train, n)
    
        print('Total train time', time.time() - start)
        
        # Save the embeddings
        print('Saving embeddings')
        self.save_embeddings(writer, trainloader)
        if testloader: self.save_embeddings(writer, testloader)
        
        # Close summary writer
        writer.close()
        
    #%%
    def _save_embeddings(self, writer, loader):
        m = len(self.model)
        N = len(loader.dataset)
        
        # Data structures for holding the embeddings
        all_data = torch.zeros(N, *self.input_shape, dtype=torch.float32, device=self.device)
        all_label = torch.zeros(N, dtype=torch.int32, device=self.device)
        all_latent = [ ]
        for j in range(m):
            all_latent.append(torch.zeros(N, self.model.latent_dim[j], dtype=torch.float32, device=self.device))
        
        # Loop over all data and get embeddings
        counter = 0
        for i, (data, label) in enumerate(loader):
            n = data.shape[0]
            data = data.reshape(-1, *self.input_shape).to(self.device)
            label = label.to(self.device)
            z = self.model.latent_representation(data)
            all_data[counter:counter+n] = data
            for j in range(m):
                all_latent[j][counter:counter+n] = z[j]
            all_label[counter:counter+n] = label
            counter += n
            
        # Save the embeddings
        for j in range(m):
            writer.add_embedding(mat = all_latent[j],
                                 metadata = all_label,
                                 label_img = all_data,
                                 tag = 'latent_space' + str(j))
            
        