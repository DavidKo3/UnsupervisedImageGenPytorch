from __future__ import print_function

import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
# for datasets ############################################
from torchvision import datasets, models, transforms
##################################################################
import time
import copy
import os
import copy

image_ind = 33
train_data = sio.loadmat('../dataset/train_32x32.mat')

# access to the dict
x_train = train_data['X']
y_train = train_data['y']

print(y_train.astype(np.int64).squeeze())
print("train_data labels :", train_data.keys())
x_train_array = np.array(x_train)
print("shape of x_train :", np.array(x_train_array).shape)
print(x_train_array[0])

x_train_rearranged = torch.from_numpy(x_train_array)
x_train_rearr_var= Variable(x_train_rearranged).permute(3,2,0,1)

print("shape of x_train_rearranged :",x_train_rearr_var.size())

length_dict = {key:len(value) for key, value in train_data.items()}
print("length_dict :", length_dict)
print("values of y : ",y_train)
print("x_train size :" ,len(x_train[0]))
print("x_train size :" ,len(x_train))
print(len(x_train[0]))
print(len(x_train[0][0]))
print(len(x_train[0][0][0]))
print(x_train[0][0][0][73256])
print(x_train[0])
print("=============================================")
print(x_train[0][0])
print("=============================================")
print()

# show sample 
plt.figure()
#plt.imshow(x_train[:,:,:, image_ind])
plt.imshow(x_train[0][31])
plt.pause(5) # pause a bit so that plots are updated
print(y_train[image_ind])
def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform = transforms.Compose([
                    transforms.Scale(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=transform)
    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform)

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    return svhn_loader, mnist_loader

use_gpu = torch.cuda.is_available()


######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
