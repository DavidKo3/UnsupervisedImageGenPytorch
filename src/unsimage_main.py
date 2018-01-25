from __future__ import print_function

import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import model

# for datasets ############################################
import torchvision
from torchvision import datasets, models, transforms
##################################################################
import time
import copy
import shutil
import os
import argparse
from dask.array.reductions import moment
from torch import FloatTensor
from bokeh.themes import default
from Cython.Compiler.PyrexTypes import best_match


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--arch', type=str, default='svhnDiscrimanator')
parser.add_argument('--startepoch', type=int, default=0)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--svhn_path', type=str, default='./datasets/svhn')
parser.add_argument('--mnist_path', type=str , default='./datasets/mnist')
parser.add_argument('--svhn_trainedmodel', type=str, default='./model_best.pth.tar')
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
    
    svhn_extra_train = datasets.SVHN(root=config.svhn_path,  split='extra', download=True, transform=transform)
    svhn_test = datasets.SVHN(root=config.svhn_path,  split='test', download=True, transform=transform)
    mnist_train = datasets.MNIST(root=config.mnist_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=config.mnist_path, train=False, transform=transform)
    
    svhn_extra_train_loader = torch.utils.data.DataLoader(dataset=svhn_extra_train,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=4)
      
    svhn_test_loader = torch.utils.data.DataLoader(dataset=svhn_test,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=4)
  
  
  
    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=4)

    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=4)
    
    return svhn_extra_train_loader,  svhn_test_loader, mnist_train_loader , mnist_test_loader




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


svhn_extra_train_loader, svhn_test_loader ,mnist_train_loader, mnist_test_loader =  get_loader(config)
mnist_iter = iter(mnist_train_loader)

print ('==>>> total trainning svhn_extra_train_loader batch number: {}'.format(len(svhn_extra_train_loader)))    # extra dataset 4 * 132,783 = 531,132
print ('==>>> total trainning svhn_test_loader batch number: {}'.format(len(svhn_test_loader)))    #

print ('==>>> total trainning mnist_train_loader batch number: {}'.format(len(mnist_train_loader)))
print ('==>>> total trainning mnist_test_loader batch number: {}'.format(len(mnist_test_loader)))



len_svhn_extra_train_loader = config.batch_size*len(svhn_extra_train_loader)
len_mnist_train_loader = config.batch_size*len(mnist_train_loader)


inputs , classes  = mnist_iter.next() # [inputs, size 4x1x32x32] , [classes size 4]
#classes = classes.view(-1)
print(" type of inputs:", type(inputs))
print("content of inputs" , (inputs))
print(" classes: ", classes)
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[classes[x] for x in [0,1,2,3]])

def save_checkpoint(state, is_best, filename='./mnist_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './mnist_model_best.pth.tar')

#def save_checkpoint(state, is_best, filename='./svhn_extra_checkpoint.pth.tar'):
#    torch.save(state, filename)
#    if is_best:
#        shutil.copyfile(filename, './svhn_extra_model_best.pth.tar')






#################################################################
# Training the model
# -----------------------
# 

#################################################################

def train_model(model, criterion, optimizer, num_epochs=5):
    since = time.time()
    
    best_model= model
    best_acc = 0.0
    if os.path.isfile(config.svhn_trainedmodel):
        print("=> loading checkpoint '{}'".format(config.svhn_trainedmodel))
        checkpoint = torch.load(config.svhn_trainedmodel)
        best_model.load_state_dict(checkpoint['state_dict']) # fixed weight , bias for network 
    
    if not os.path.isfile(config.svhn_trainedmodel):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('--------------------------------------------')
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    #model.train(True)
                else:
                    model.eval()
                    # model.train(False)
                    
                running_loss =0.0
                running_corrects = 0
                
                #Iterate over data.
                for data in svhn_extra_train_loader:
                    # get the inputs
                    inputs, labels = data
                    # print(model)
                    # wrap them in Variable
                    #if use_gpu:
                    # intpus = model.to_var(inputs)
                    # inputs = inputs.type(torch.FloatTensor),
                    labels = labels.type(torch.LongTensor)       
                    labels = labels.long().squeeze() # from svhan labels byte to long and reform size 4x1 to size 4      
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    
                    #else:
                    #    inputs, labels = Variable(inputs), Variable(labels)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                   
                    # forward 
                    outputs = model(inputs)
                    # print("-----------------outputs------------------------")
                    # print(outputs)  # FloatTensor of size 4x10
                    # print("-----------------labels------------------------")
                    # print(labels)   # LongTensor of size 4
                    _ , preds = torch.max(outputs.data , 1) # max index with row
                    # print("-----------------prediction--------------------")
                    # print(preds)    # LongTensor of size 4 
                    
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                       #  print("loss" , loss)
                    # statistics
                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds == labels.data)
                    
                # statistics
                epoch_loss = running_loss/len_svhn_extra_train_loader   
                epoch_acc = running_corrects /len_svhn_extra_train_loader 
                
                print('{} Loss : {:.4f} Acc : {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
                # deep copy yhe model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    is_best = best_acc
                    #
                    # best_model = copy.deepcopy(model)
                    save_checkpoint({'epoch': epoch+1 , 'arch': config.arch, 'state_dict' : model.state_dict(), 'best_acc': best_acc},is_best)
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

use_gpu = torch.cuda.is_available()
    
if torch.cuda.is_available() :
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")     
         
#print("11")
model_ft = model.D2().cuda()
#print("22")


criterion = nn.CrossEntropyLoss()         
optimzer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

       
# Train and Evaluate
#print("33")
model_ft = train_model(model_ft, criterion, optimzer_ft, num_epochs=5)    
   
#print("44")
                
best_prec_SVHN= 0

if os.path.isfile(config.svhn_trainedmodel):
    print("=> loading checkpoint '{}'".format(config.svhn_trainedmodel))
    checkpoint = torch.load(config.svhn_trainedmodel)
   #  model_ft.load_state_dict(checkpoint['state_dict'])
    best_prec_SVHN = checkpoint['best_acc']
   # print(config.arch)
   # print(checkpoint['arch'])
   # print(best_prec_SVHN)
   # print(model_ft.state_dict())
   # print("dsfsdfs")