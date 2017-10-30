from __future__ import print_function

import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import copy

image_ind = 33
train_data = sio.loadmat('../dataset/train_32x32.mat')

# access to the dict
x_train = train_data['X']
y_train = train_data['y']
print("train_data labels :", train_data.keys())
length_dict = {key:len(value) for key, value in train_data.items()}
print("length_dict :", length_dict)
print("x_train size :" ,len(x_train[0]))
print("x_train size :" ,len(x_train))
print(len(x_train[0]))
print(len(x_train[0][0]))
print(len(x_train[0][0][0]))
print(x_train[0][0][0][73256])
# show sample 
plt.figure()

plt.imshow(x_train[:,:,:, image_ind])
plt.pause(5) # pause a bit so that plots are updated
print(y_train[image_ind])


