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
from torchvision.utils import make_grid
from tqdm import tqdm
import os, datetime, time
from tensorboardX import SummaryWriter

#%%
class args:
    model = 'vae-stn'
    batch_size = 128
    input_shape = (1, 28, 28)
    n_epochs = 500
    lr = 1e-4
    use_cuda = True
    warmup = 200
    latent_dim = 20
    only_ones = True
    n_show = 10

#%%
class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, intermidian_size=400):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.flat_dim = np.prod(input_shape)
        self.fc1 = nn.Linear(self.flat_dim, intermidian_size)
        self.fc2 = nn.Linear(intermidian_size, self.latent_dim)
        self.fc3 = nn.Linear(intermidian_size, self.latent_dim)
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = x.view(-1, self.flat_dim)
        h = self.activation(self.fc1(x))
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
        self.activation = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(self.latent_dim, intermidian_size)
        self.fc2 = nn.Linear(intermidian_size, self.flat_dim)
        
    def forward(self, x):
        h  = self.activation(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out.view(-1, *self.output_shape)

#%%
class NewDecoder(nn.Module):
    def __init__(self, output_shape, latent_dim, intermidian_size=400):
        super(NewDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.flat_dim = np.prod(output_shape)
        self.activation = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(self.latent_dim, intermidian_size)
        self.fc2 = nn.Linear(intermidian_size, self.flat_dim)
        self.fc2.weight = torch.nn.Parameter(torch.zeros_like(self.fc2.weight))
        self.fc2.bias = torch.nn.Parameter(torch.tensor([1.0,0,0,0,1.0,0]))
        
    def forward(self, x):
        h = self.activation(self.fc1(x))
        out = self.activation(self.fc2(h))
        return out.view(-1, *self.output_shape)
        
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
        return out, mu1, logvar1, mu2, logvar2
    
    def sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z1 = torch.randn(n, self.encoder1.latent_dim, device=device)
            z2 = torch.randn(n, self.encoder2.latent_dim, device=device)
            dec = self.decoder1(z1)
            theta = self.decoder2(z2)
            out = self.stn(dec, theta)
            return out
    
    def sample_only_images(self, n, trans):
        device = next(self.parameters()).device
        with torch.no_grad():
            trans = trans[None, :].repeat(n, 1).to(device)
            z1 = torch.randn(n, self.encoder1.latent_dim, device=device)
            dec = self.decoder1(z1)
            out = self.stn(dec, trans)
            return out
        
    def sample_only_trans(self, n, img):
        device = next(self.parameters()).device
        with torch.no_grad():
            img = img.repeat(n, 1, 1, 1).to(device)
            z2 = torch.randn(n, self.encoder2.latent_dim, device=device)
            theta = self.decoder2(z2)
            out = self.stn(img, theta)
            return out
    
    def latent_representation(self, x):
        mu1, logvar1 = self.encoder1(x)
        mu2, logvar2 = self.encoder2(x)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        return z1, z2
    
    def sample_transformation(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z2 = torch.randn(n, self.encoder2.latent_dim, device=device)
            theta = self.decoder2(z2)
            return theta
   
#%%
def reconstruction_loss(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return BCE

#%%
def kullback_leibler_divergence(mu, logvar, epoch=1, warmup=1):
    scaling = np.min([epoch / warmup, 1])
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD *= scaling
    return KLD

#%%
if __name__ == '__main__':
    logdir = args.model + '/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    if not os.path.exists(logdir): os.makedirs(logdir)
    
    # Load data
    if args.only_ones:
        from mnist_only_ones import MNIST_only_ones
        train = MNIST_only_ones(root='', transform=transforms.ToTensor(), download=True)
        test = MNIST_only_ones(root='', train=False, transform=transforms.ToTensor(), download=True)
    else:
        train = datasets.MNIST(root='', transform=transforms.ToTensor(), download=True)
        test = datasets.MNIST(root='', train=False, transform=transforms.ToTensor(), download=True)
    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size)
    testloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size)
    
    # Summary writer
    writer = SummaryWriter(log_dir=logdir)
    
    # Save device
    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
        
    # Construct model
    encoder1 = Encoder(input_shape=args.input_shape, latent_dim=args.latent_dim, intermidian_size=400)
    encoder2 = Encoder(input_shape=args.input_shape, latent_dim=args.latent_dim, intermidian_size=400)
    decoder1 = Decoder(output_shape=args.input_shape, latent_dim=args.latent_dim, intermidian_size=400)
    decoder2 = NewDecoder(output_shape=(6,), latent_dim=args.latent_dim, intermidian_size=10)
    
    stn = STN(input_shape=args.input_shape)
    model = VAE_with_STN(encoder1, encoder2, decoder1, decoder2, stn)
    
    # Move model to gpu
    if torch.cuda.is_available() and args.use_cuda:
        model.cuda()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Main loop
    start = time.time()
    for epoch in range(1, args.n_epochs+1):
        progress_bar = tqdm(desc='Epoch ' + str(epoch), total=len(trainloader.dataset), 
                                unit='samples')
        train_loss = 0    
        # Training loop
        model.train()
        for i, (data, _) in enumerate(trainloader):
            # Zero gradient
            optimizer.zero_grad()
            
            # Feed forward data
            data = data.reshape(-1, *args.input_shape).to(device)
            recon_data, mu1, logvar1, mu2, logvar2 = model(data)    
            
            # Calculat loss
            recon_loss = reconstruction_loss(recon_data, data)
            kl1_loss = kullback_leibler_divergence(mu1, logvar1, epoch, args.warmup)
            kl2_loss = kullback_leibler_divergence(mu2, logvar2, epoch, args.warmup)
            loss = recon_loss + kl1_loss + kl2_loss
            train_loss += loss.item()
            
            # Backpropegate and optimize
            loss.backward()
            optimizer.step()
            
            # Write to consol and tensorboard
            progress_bar.update(data.size(0))
            progress_bar.set_postfix({'loss': loss.item()})
            iteration = epoch*len(trainloader) + i
            writer.add_scalar('train/total_loss', loss, iteration)
            writer.add_scalar('train/recon_loss', recon_loss, iteration)
            writer.add_scalar('train/KL_loss1', kl1_loss, iteration)
            writer.add_scalar('train/KL_loss2', kl2_loss, iteration)
            
        progress_bar.set_postfix({'Average loss': train_loss / len(trainloader)})
        progress_bar.close()
        
        # Try out on test data
        model.eval()
        recon_loss
        for i, (data, _) in enumerate(testloader):
            data = data.reshape(-1, *args.input_shape).to(device)
            recon_data, mu1, logvar1, mu2, logvar2 = model(data)    
            
            recon_loss += reconstruction_loss(recon_data, data)
            kl1_loss += kullback_leibler_divergence(mu1, logvar1, epoch, args.warmup)
            kl2_loss += kullback_leibler_divergence(mu2, logvar2, epoch, args.warmup)
        test_loss = recon_loss + kl1_loss + kl2_loss
        
        writer.add_scalar('test/total_loss', test_loss, iteration)
        writer.add_scalar('test/recon_loss', recon_loss, iteration)
        writer.add_scalar('test/KL_loss1', kl1_loss, iteration)
        writer.add_scalar('test/KL_loss2', kl2_loss, iteration)
        
        # Save reconstructions to tensorboard
        n = args.n_show
        data_train = next(iter(trainloader))[0].to(device)[:n]
        data_test = next(iter(testloader))[0].to(device)[:n]
        recon_data_train = model(data_train)[0]
        recon_data_test = model(data_test)[0]
        
        writer.add_image('train/recon', make_grid(torch.cat([data_train, 
                         recon_data_train]).cpu(), nrow=n), global_step=epoch)
        writer.add_image('test/recon', make_grid(torch.cat([data_test, 
                         recon_data_test]).cpu(), nrow=n), global_step=epoch)
        
        # Save sample to tensorboard
        samples = model.sample(n*n)    
        writer.add_image('samples/samples', make_grid(samples.cpu(), nrow=n), 
                         global_step=epoch)
        
        trans = torch.tensor([1.0,0,0,0,1.0,0])
        samples = model.sample_only_images(n*n, trans)
        writer.add_image('samples/fixed_trans', make_grid(samples.cpu(), nrow=n),
                         global_step=epoch)
        
        img = data_train[0]
        samples = model.sample_only_trans(n*n, img)
        writer.add_image('samples/fixed_img', make_grid(samples.cpu(), nrow=n),
                         global_step=epoch)
        
        # Lets log a histogram of the transformation
        theta = model.sample_transformation(1000)
        for i in range(6):
            writer.add_histogram('transformation/a' + str(i), theta[:,i], global_step=epoch)
        
        # Lets plot the mean of the transformation
        writer.add_image('transformation/mean', global_step=epoch, 
                         img_tensor=stn(img[None], theta.mean(dim=0, keepdim=True)))
        
    print('Total train time', time.time() - start)
    
    # Save some embeddings
    print('Saving embeddings')
    all_data = torch.zeros(len(test), *args.input_shape, dtype=torch.float32, device=device)
    all_latent1 = torch.zeros(len(test), args.latent_dim, dtype=torch.float32, device=device)
    all_latent2 = torch.zeros(len(test), args.latent_dim, dtype=torch.float32, device=device)
    all_label = torch.zeros(len(test), dtype=torch.int32, device=device)
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