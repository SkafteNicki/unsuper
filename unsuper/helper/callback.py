# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 12:25:15 2018

@author: Nicki
"""
#%%
import torch
from torchvision.utils import make_grid

#%%
def callback_vitae(writer, model, epoch, data_train, n):
    trans = torch.tensor([0,0,0,0,0,0], dtype=torch.float32)
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
    
    # Lets plot the mean of the transformation for a couple of images
    for i in range(4):
        writer.add_image('transformation/mean' + str(i), global_step=epoch, 
                         img_tensor=stn(data_train[i][None], theta.mean(dim=0, keepdim=True)))
