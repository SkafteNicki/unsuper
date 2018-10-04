# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 09:01:59 2018

@author: nsde
"""

#%%
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os, datetime
from tensorboardX import SummaryWriter
   
#%%
class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, intermidian_size=400):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.flat_dim = np.prod(input_shape)
        self.fc1 = nn.Linear(self.flat_dim, intermidian_size)
        self.fc2 = nn.Linear(intermidian_size, self.latent_dim)
        self.fc3 = nn.Linear(intermidian_size, self.latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, self.flat_dim)
        h = self.relu(self.fc1(x))
        mu = self.fc2(h)
        logvar = self.fc3(h)
        return mu, logvar
   
#%%     
class Decoder(nn.Module):
    def __init__(self, output_shape, latent_dim, intermidian_size=400):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.flat_dim = np.prod(output_shape)
        self.fc1 = nn.Linear(self.latent_dim, intermidian_size)
        self.fc2 = nn.Linear(intermidian_size, self.flat_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h  = self.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out.view(-1, *self.output_shape)

#%%
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), (mu, logvar)
    
    def sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(n, self.decoder.latent_dim, device=device)
            return self.decoder(z)
            
    def latent_representation(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

#%%
class STN(nn.Module):
    def __init__(self, input_shape):
        super(STN, self).__init__()
        self.input_shape = input_shape
        
    def forward(self, x, theta):
        theta = theta.view(-1, 2, 3)
        output_size = torch.Size([x.shape[0], *self.input_shape])
        grid = F.affine_grid(theta, output_size)
        x = F.grid_sample(x, grid)
        return x

#%%
class VAE_with_STN(nn.Module):
    def __init__(self, encoder1, encoder2, decoder1, decoder2, stn):
        super(VAE_with_STN, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.stn = stn
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu
        
    def forward(self, x):
        mu1, logvar1 = self.encoder1(x)
        mu2, logvar2 = self.encoder2(x)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        dec = self.decoder1(z1)
        theta = self.decoder2(z2)
        out = self.stn(dec, theta)
        return out, (mu1, logvar1, mu2, logvar2)
    
    def sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z1 = torch.randn(n, self.encoder1.latent_dim, device=device)
            z2 = torch.randn(n, self.encoder2.latent_dim, device=device)
            dec = self.decoder1(z1)
            theta = self.decoder2(z2)
            out = self.stn(dec, theta)
            return out
        
    def latent_representation(self, x):
        mu1, logvar1 = self.encoder1(x)
        mu2, logvar2 = self.encoder2(x)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        return z1, z2
            
#%%
def loss_function(kl_scaling, recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if kl_scaling: KLD /= recon_x.numel() # normalize KL divergence
    return BCE+KLD, (BCE, KLD)

#%%
def loss_function_with_stn(kl_scaling, recon_x, x, mu1, logvar1, mu2, logvar2):
    BCE = F.binary_cross_entropy(recon_x, x)
    KLD1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    if kl_scaling: KLD1 /= recon_x.numel()
    KLD2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    if kl_scaling: KLD2 /= recon_x.numel()
    return BCE + KLD1 + KLD2, (BCE, KLD1, KLD2)

#%%
class args:
    model = 'vae-stn'
    batch_size = 128
    input_shape = (1, 28, 28)
    n_epochs = 100
    lr = 1e-4
    use_cuda = True
    kl_scaling = True

#%%
if __name__ == '__main__':
    logdir = args.model + '/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    if not os.path.exists(logdir): os.makedirs(logdir)
    # Load data
    train = datasets.MNIST(root='', transform=transforms.ToTensor(), download=True)
    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size)
    test = datasets.MNIST(root='', train=False, transform=transforms.ToTensor(), download=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size)
    
    # Summary writer
    writer = SummaryWriter(log_dir=logdir)
    
    # Save device
    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Construct model
    if args.model == 'vae':
        encoder = Encoder(input_shape=args.input_shape, latent_dim=20, intermidian_size=400)
        decoder = Decoder(output_shape=args.input_shape, latent_dim=20, intermidian_size=400)
        model = VAE(encoder, decoder)
        loss_f = loss_function
    elif args.model == 'vae-stn':
        encoder1 = Encoder(input_shape=args.input_shape, latent_dim=20, intermidian_size=400)
        encoder2 = Encoder(input_shape=args.input_shape, latent_dim=2, intermidian_size=400)
        decoder1 = Decoder(output_shape=args.input_shape, latent_dim=20, intermidian_size=400)
        decoder2 = Decoder(output_shape=(6,), latent_dim=2, intermidian_size=4)
        stn = STN(input_shape=args.input_shape)
        model = VAE_with_STN(encoder1, encoder2, decoder1, decoder2, stn)
        loss_f = loss_function_with_stn
    
    # Save graph
#    writer.add_graph(model=model, 
#                     input_to_model=torch.autograd.Variable(next(iter(trainloader))[0]))
    
    # Move model to gpu
    if torch.cuda.is_available() and args.use_cuda:
        model.cuda()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Main loop
    for epoch in range(1, args.n_epochs+1):
        progress_bar = tqdm(desc='Epoch ' + str(epoch), total=len(trainloader.dataset), 
                                unit='samples')
        train_loss = 0    
        # Training loop
        model.train()
        for i, (data, _) in enumerate(trainloader):
            optimizer.zero_grad()
            data = data.reshape(-1, *args.input_shape).to(device)
            recon_data, latents = model(data)    
            loss, individual_loss = loss_f(args.kl_scaling, recon_data, data, *latents)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            progress_bar.update(data.size(0))
            progress_bar.set_postfix({'loss': loss.item()})
            writer.add_scalar('train/total_loss', loss, epoch*len(trainloader) + i)
            writer.add_scalar('train/recon_loss', individual_loss[0], epoch*len(trainloader) + i)
            writer.add_scalar('train/KL_loss', sum(individual_loss[1:]), epoch*len(trainloader) + i)
            
        progress_bar.set_postfix({'Average loss': train_loss / len(trainloader)})
        progress_bar.close()
        
        # Try out on test data
        model.eval()
        test_loss, recon_loss, KL_loss = 0, 0, 0
        for i, (data, _) in enumerate(testloader):
            data = data.reshape(-1, *args.input_shape).to(device)
            recon_data, latents = model(data)
            loss, individual_loss = loss_f(args.kl_scaling, recon_data, data, *latents)
            test_loss += loss
            recon_loss += individual_loss[0]
            KL_loss += sum(individual_loss[1:])
            
        writer.add_scalar('test/total_loss', test_loss, epoch*len(trainloader) + i)
        writer.add_scalar('test/recon_loss', recon_loss, epoch*len(trainloader) + i)
        writer.add_scalar('test/KL_loss', KL_loss, epoch*len(trainloader) + i)
        
        
        
        # Save some results
        n = 15
        samples = model.sample(n)
        data_train = next(iter(trainloader))[0].to(device)
        data_test = next(iter(testloader))[0].to(device)
        comparison = torch.cat([data_train[:n], model(data_train[:n])[0],
                                data_test[:n], model(data_test[:n])[0],
                                samples])
        save_image(comparison.cpu(), logdir + '/samp_recon' + str(epoch) + '.png', nrow=n)
    
    # TODO: generalize this
    print('Saving embeddings')
    if args.model == 'vae':
        all_data = torch.zeros(10000, 1, 28, 28, dtype=torch.float32, device=device)
        all_latent = torch.zeros(10000, 20, dtype=torch.float32, device=device)
        all_label = torch.zeros(10000, dtype=torch.int32, device=device)
        counter = 0
        for i, (data, label) in enumerate(testloader):
            n = data.shape[0]
            data = data.reshape(-1, *args.input_shape).to(device)
            label = label.to(device)
            z, _ = encoder(data)
            all_data[counter:counter+n] = data
            all_latent[counter:counter+n] = z
            all_label[counter:counter+n] = label
            counter += n
        writer.add_embedding(mat = all_latent,
                             metadata= all_label,
                             label_img= all_data,
                             tag = 'latent_space')
    
    elif args.model == 'vae-stn':
        all_data = torch.zeros(10000, 1, 28, 28, dtype=torch.float32, device=device)
        all_latent1 = torch.zeros(10000, 20, dtype=torch.float32, device=device)
        all_latent2 = torch.zeros(10000, 2, dtype=torch.float32, device=device)
        all_label = torch.zeros(10000, dtype=torch.int32, device=device)
        counter = 0
        for i, (data, label) in enumerate(testloader):
            n = data.shape[0]
            data = data.reshape(-1, *args.input_shape).to(device)
            label = label.to(device)
            z1, _ = encoder1(data)
            z2, _ = encoder2(data)
            all_data[counter:counter+n] = data
            all_latent1[counter:counter+n] = z1
            all_latent2[counter:counter+n] = z2
            all_label[counter:counter+n] = label
            counter += n
            
        writer.add_embedding(mat = all_latent1,
                             metadata = all_label,
                             label_img = all_data,
                             tag = 'latent_space1')
        writer.add_embedding(mat = all_latent2,
                             metadata = all_label,
                             label_img = all_data,
                             tag = 'latent_space2')
    writer.close()