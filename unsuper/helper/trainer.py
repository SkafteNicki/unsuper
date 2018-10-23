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
from .losses import ELBO

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
            testloader=None):
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
            # Training loop
            self.model.train()
            for i, (data, _) in enumerate(trainloader):
                # Zero gradient
                self.optimizer.zero_grad()
            
                # Feed forward data
                data = data.reshape(-1, *self.input_shape).to(self.device)
                recon_data, mus, logvars = self.model(data)
                
                # Calculat loss
                loss, recon_term, kl_terms = ELBO(data, recon_data, mus, 
                                                  logvars, epoch, warmup)
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
                with torch.no_grad():
                    # Evaluate on test set
                    self.model.eval()
                    test_loss, test_recon, test_kl = 0, 0, len(kl_terms)*[0]
                    for i, (data, _) in enumerate(testloader):
                        data = data.reshape(-1, *self.input_shape).to(self.device)
                        recon_data, mus, logvars = self.model(data)    
                        loss, recon_term, kl_terms = ELBO(data, recon_data, mus, 
                                                          logvars, epoch, warmup)
                        test_loss += loss.item()
                        test_recon += recon_term.item()
                        test_kl = [l1+l2 for l1,l2 in zip(kl_terms, test_kl)]
            
                    writer.add_scalar('test/total_loss', test_loss, iteration)
                    writer.add_scalar('test/recon_loss', recon_term, iteration)
                    for j, kl_loss in enumerate(kl_terms):
                        writer.add_scalar('test/KL_loss' + str(j), kl_loss, iteration)
            
                    data_test = next(iter(testloader))[0].to(self.device)[:n]
                    recon_data_test = self.model(data_test)[0]
                    writer.add_image('test/recon', make_grid(torch.cat([data_test, 
                             recon_data_test]).cpu(), nrow=n), global_step=epoch)
                    
                    # If callback call it now
                    self.model.callback(writer, testloader, epoch)
    
        print('Total train time', time.time() - start)
        
        # Save the embeddings
        print('Saving embeddings')
        with torch.no_grad():
            self._save_embeddings(writer, trainloader, name='train')
            if testloader: self._save_embeddings(writer, testloader, name='test')
        
        # Compute marginal log likelihood on the test set
        if testloader:
            logp = self.eval_log_prob(testloader, 10000)
            print('Marginal log likelihood:', logp)
            writer.add_text('Test marginal log likelihood',  str(logp))
        
        # Close summary writer
        writer.close()
        
    #%%
    def _save_embeddings(self, writer, loader, name='embedding'):
        m = len(self.model)
        N = len(loader.dataset)
        
        # Data structures for holding the embeddings
        all_data = torch.zeros(N, *self.input_shape, dtype=torch.float32)
        all_label = torch.zeros(N, dtype=torch.int32)
        all_latent = [ ]
        for j in range(m):
            all_latent.append(torch.zeros(N, self.model.latent_dim[j], dtype=torch.float32))
        
        # Loop over all data and get embeddings
        counter = 0
        for i, (data, label) in enumerate(loader):
            n = data.shape[0]
            data = data.reshape(-1, *self.input_shape).to(self.device)
            label = label.to(self.device)
            z = self.model.latent_representation(data)
            all_data[counter:counter+n] = data.cpu()
            for j in range(m):
                all_latent[j][counter:counter+n] = z[j].cpu()
            all_label[counter:counter+n] = label.cpu()
            counter += n
            
        # Save the embeddings
        for j in range(m):
            # Maximum bound for the sprite image
            if all_data.shape[0] * all_data.shape[2] * all_data.shape[3] < 8192:
                writer.add_embedding(mat = all_latent[j],
                                     metadata = all_label,
                                     label_img = all_data,
                                     tag = name + '_latent_space' + str(j))
            else:
                writer.add_embedding(mat = all_latent[j],
                                     metadata = all_label,
                                     tag = name + '_latent_space' + str(j))
                
    #%%
    def eval_log_prob(self, testloader, S):
        means = self.model.sample(S)
        means = means.view(S, -1)
        cov = torch.eye(means.shape[1])
        cov = cov.repeat(S, 1, 1)
        distribution = torch.distributions.MultivariateNormal(loc=means,
                                                              covariance_matrix=cov)
        total_logp = 0
        for i, (data, _) in enumerate(testloader):
            data_flat = data.view(S, -1)
            logp = distribution.log_prob(data_flat)
            total_logp += logp.mean().item()
        total_logp /= len(testloader)
        return total_logp
            
        
        
        