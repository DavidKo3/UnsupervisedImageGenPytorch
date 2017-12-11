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
import torchvision
from torchvision import datasets, models, transforms
##################################################################
import time
import copy
import os
import copy
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--svhn_path', type=str, default='./datasets/svhn')
parser.add_argument('--mnist-path', type=str , default='./datasets/mnist')
config = parser.parse_args()
print(config)
print(config.mnist_path)


image_ind = 33
train_data = sio.loadmat('../datasets/svhn/train_32x32.mat')

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
    
    svhn_train = datasets.SVHN(root=config.svhn_path,  download=True, transform=transform)
   
    mnist_train = datasets.MNIST(root=config.mnist_path, train=True,download=True, transform=transform)
    mnist_test = datasets.MNIST(root=config.mnist_path, train=False, transform=transform)
    
    svhn_train_loader = torch.utils.data.DataLoader(dataset=svhn_train,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=4)
  
    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=4)

    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=4)
    
    return svhn_train_loader,  mnist_train_loader , mnist_test_loader

use_gpu = torch.cuda.is_available()


######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    print("sdfsdf")
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving (deep copying) the best model
#
# In the following, parameter ``lr_scheduler(optimizer, epoch)``
# is a function  which modifies ``optimizer`` so that the learning
# rate is changed according to desired schedule.

def to_var(x):
    """Converts numpy to variable"""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """"Converts variable to numpy"""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


svhn_train_loader, mnist_train_loader, mnist_test_loader =  get_loader(config)
mnist_iter = iter(mnist_train_loader)


inputs , classes  = mnist_iter.next() # [inputs, size 4x1x32x32] , [classes size 4]
print(" inputs:", inputs)
print(" classes: ", classes)
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[classes[x] for x in [0,1,2,3]])









#################################################################
# Training the model
# -----------------------
# 

#################################################################

def train_model(model, criterion, optimizer, num_epochs=5):
    since = time.time()
    
    best_model= model
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('--------------------------------------------')
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
                
            running_loss =0.0
            running_corrects = 0
            
            #Iterate over data.
            for data in mnist_train_loader:
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                


    