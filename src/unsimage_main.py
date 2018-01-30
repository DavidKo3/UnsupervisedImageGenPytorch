from __future__ import print_function

import numpy as np
import scipy.misc 
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

"""
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
"""

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
len_svhn_test_loader = config.batch_size*len(svhn_test_loader)
len_mnist_train_loader = config.batch_size*len(mnist_train_loader)
len_mnist_test_loader = config.batch_size*len(mnist_test_loader)


data_loader = {'train': svhn_extra_train_loader, 'val' :svhn_test_loader}
len_data_loader = {'train': len_svhn_extra_train_loader, 'val' : len_svhn_test_loader}
inputs , classes  = mnist_iter.next() # [inputs, size 4x1x32x32] , [classes size 4]
#classes = classes.view(-1)
#print(" type of inputs:", type(inputs))
#print("content of inputs" , (inputs))
# print(" classes: ", classes)
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[classes[x] for x in [0,1,2,3]])


def merge_images(sources, targets, k=10):
    _, _, h, w = sources.shape
    row = int(np.sqrt(config.batch_size))
    merged = np.zeros([3, row*h, row*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
    return merged.transpose(1, 2, 0)
   
def to_var(self, x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
    
def to_data(self, x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
            x = x.cpu()
    return x.data.numpy()


def save_checkpoint(state, is_best, filename='./svhn_extra_mnist_model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './svhn_extra_mnist_checkpoint.pth.tar')

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
                for data in data_loader[phase]:
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
                
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    is_best = best_acc
                    #
                    # best_model = copy.deepcopy(model)
                    # save_checkpoint({'epoch': epoch+1 , 'arch': config.arch, 'state_dict' : model.state_dict(), 'best_acc': best_acc},is_best)
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

def test_model(model, criterion, optimizer, num_epochs=1000):
    since = time.time()
    
    best_model= model
    best_acc = 0.0
    if os.path.isfile(config.svhn_trainedmodel):
        print("=> loading checkpoint '{}'".format(config.svhn_trainedmodel))
        checkpoint = torch.load(config.svhn_trainedmodel)
        best_model.load_state_dict(checkpoint['state_dict']) # fixed weight , bias for network 
    
    if not os.path.isfile(config.svhn_trainedmodel):
        
        test_loss =0.0
        test_corrects = 0
            
        #Iterate over data.
        for data in data_loader['val']:
            # get the inputs
            inputs, labels = data
                # print(model)
                # wrap them in Variable
                #if use_gpu:
                # intpus = model.to_var(inputs)
                # inputs = inputs.type(torch.FloatTensor),
            labels = labels.type(torch.LongTensor)       
            labels = labels.long().squeeze() # from svhan labels byte to long and reform size 4x1 to size 4      
            inputs, labels = Variable(inputs.cuda(),  volatile=True), Variable(labels.cuda())
                   
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


            # statistics
            test_loss += loss.data[0]
            test_corrects += torch.sum(preds == labels.data)
           # print('{} len_svhn_test:{:.4f}, loss :{:.4f}, acc:{:.4f}'.format(epoch, len_data_loader['val'] , test_loss, test_corrects))
                 
        # statistics
        test_loss = test_loss/len_data_loader['val']   
        test_corrects = test_corrects /len_data_loader['val']
                
        print('{} Loss : {:.4f} Acc : {:.4f}'.format('test', test_loss, test_corrects))
                
        # deep copy the model
        if test_corrects > best_acc:
            best_acc = test_corrects
            is_best = best_acc
                #
                # best_model = copy.deepcopy(model)
                # save_checkpoint({'epoch': epoch+1 , 'arch': config.arch, 'state_dict' : model.state_dict(), 'best_acc': best_acc},is_best)
    print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

def train_test_model(model, criterion, optimizer, num_epochs=3):
    since = time.time()
    
    best_model= model
    best_acc = 0.0
    if os.path.isfile(config.svhn_trainedmodel):
        print("=> loading checkpoint '{}'".format(config.svhn_trainedmodel))
        checkpoint = torch.load(config.svhn_trainedmodel)
        best_model.load_state_dict(checkpoint['state_dict']) # fixed weight , bias for network 
    
    if not os.path.isfile(config.svhn_trainedmodel):
        
        training_loss =0.0
        training_corrects = 0
        
        test_loss =0.0
        test_corrects = 0
        
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
                    
                #Iterate over data.
                for data in data_loader[phase]:
                    # get the inputs
                    inputs, labels = data
                    # print(model)
                    # wrap them in Variable
                    #if use_gpu:
                    # intpus = model.to_var(inputs)
                    # inputs = inputs.type(torch.FloatTensor),
                    labels = labels.type(torch.LongTensor)       
                    labels = labels.long().squeeze() # from svhan labels byte to long and reform size 4x1 to size 4      
                    if phase == 'train':
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        _ , preds = torch.max(outputs.data , 1) # max index with row
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        # statistics
                        training_loss = loss.data[0]
                        training_corrects += torch.sum(preds == labels.data)
                    else:
                        inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda())
                        outputs = model(inputs)
                        _ , preds = torch.max(outputs.data , 1) # max index with row
                        loss = criterion(outputs, labels)
                        test_loss += loss.data[0]
                        test_corrects += torch.sum(preds == labels.data)
                        print("\ntest_corrects : ", test_corrects)
            
                        
                # statistics
                epoch_training_loss = training_loss/len_data_loader['train']   
                epoch_training_acc = training_corrects/len_data_loader['train']   
                epoch_test_loss = test_loss/len_data_loader['val']     
                epoch_test_acc = test_corrects/len_data_loader['val']  
                if phase =='train':
                    print('\n len of svhn train dataset :{}'.format(len_data_loader['train']))
                    print('\n{} Loss : {:.8f} Acc : {:.4f}'.format(phase, epoch_training_loss, epoch_training_acc))
                else:
                    print('\n len of svhn test dataset :{}'.format(len_data_loader['val'] ))
                    print('\n{} Loss : {:.4f} Acc : {:.4f}'.format(phase, epoch_test_loss, epoch_test_acc))
                # deep copy the model
                if phase == 'val' and epoch_test_acc > best_acc:
                    best_acc = epoch_test_acc
                    is_best = best_acc
                    #
                    # best_model = copy.deepcopy(model)
                    # save_checkpoint({'epoch': epoch+1 , 'arch': config.arch, 'state_dict' : model.state_dict(), 'best_acc': best_acc},is_best)
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val test Acc: {:4f}'.format(best_acc))
    return best_model


def train_model_v2(model, criterion, optimizer, num_epochs):
    since = time.time()
    
    best_model= model
    best_acc = 0.0
    if os.path.isfile(config.svhn_trainedmodel):
        print("=> loading checkpoint '{}'".format(config.svhn_trainedmodel))
        checkpoint = torch.load(config.svhn_trainedmodel)
        best_model.load_state_dict(checkpoint['state_dict']) # fixed weight , bias for network 
    
    if not os.path.isfile(config.svhn_trainedmodel):
        
        print('\nTrain Epoch {}'.format(num_epochs))
        print('--------------------------------------------')
            
        # Each epoch has a training and validation phase
        
        model.train()
        #model.train(True)

        running_loss =0.0
        running_corrects = 0
                
        #Iterate over data.
        for data in data_loader['train']:
            # get the inputs
            inputs, labels = data
            labels = labels.type(torch.LongTensor)       
            labels = labels.long().squeeze() # from svhan labels byte to long and reform size 4x1 to size 4      
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            # zero the parameter gradients
            optimizer.zero_grad()       
            # forward 
            outputs = model(inputs)

            _ , preds = torch.max(outputs.data , 1) # max index with row

                    
            loss = criterion(outputs, labels)        
            loss.backward()
            optimizer.step()

            running_loss = loss.data[0]
            running_corrects += torch.sum(preds == labels.data)
                    
        # statistics
        epoch_loss = running_loss/len_svhn_extra_train_loader   
        epoch_acc = running_corrects /len_svhn_extra_train_loader 
                
        print('\nTrain Loss : {:.6f} Acc : {:.4f}'.format(epoch_loss, epoch_acc))
                
        # deep copy the model
        if  epoch_acc > best_acc:
            best_acc = epoch_acc
            is_best = best_acc
            best_model = copy.deepcopy(model)
            # save_checkpoint({'epoch': epoch+1 , 'arch': config.arch, 'state_dict' : model.state_dict(), 'best_acc': best_acc},is_best)
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

def test_model_v2(model, criterion, optimizer, num_epochs):
    since = time.time()
    
    best_model= model
    best_acc = 0.0
    if os.path.isfile(config.svhn_trainedmodel):
        print("=> loading checkpoint '{}'".format(config.svhn_trainedmodel))
        checkpoint = torch.load(config.svhn_trainedmodel)
        best_model.load_state_dict(checkpoint['state_dict']) # fixed weight , bias for network 
    
    if not os.path.isfile(config.svhn_trainedmodel):
        print('\nTest Epoch {}'.format(num_epochs))
        print('--------------------------------------------')
        test_loss =0.0
        test_corrects = 0
            
        #Iterate over data.
        for data in data_loader['val']:
            # get the inputs
            inputs, labels = data
            labels = labels.type(torch.LongTensor)       
            labels = labels.long().squeeze() # from svhan labels byte to long and reform size 4x1 to size 4      
            inputs, labels = Variable(inputs.cuda(),  volatile=True), Variable(labels.cuda())
                   
            # forward 
            outputs = model(inputs)
            _ , preds = torch.max(outputs.data , 1) # max index with row                
            loss = criterion(outputs, labels)


            # statistics
            test_loss += loss.data[0]
            test_corrects += torch.sum(preds == labels.data)
           # print('{} len_svhn_test:{:.4f}, loss :{:.4f}, acc:{:.4f}'.format(epoch, len_data_loader['val'] , test_loss, test_corrects))
                 
        # statistics
        test_loss = test_loss/len_data_loader['val']   
        test_corrects = test_corrects /len_data_loader['val']
                
        print('Test Loss : {:.4f} Acc : {:.4f}'.format(test_loss, test_corrects))
                
        # deep copy the model
        if test_corrects > 0.9050:
            best_acc = test_corrects
            is_best = best_acc
            best_model = copy.deepcopy(model)
            save_checkpoint({'epoch': epoch+1 , 'arch': config.arch, 'state_dict' : model.state_dict(), 'best_acc': best_acc},is_best)
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

def train_generated_model_(model_generator, model_dicriminator1, model_dicriminator2 , model_dicriminator3, criterion, optimizer_g, optimizer_d1, optimizer_d2,optimizer_d3,num_epochs):
    since = time.time()
    
    # trained_svhn_dicscrimator_model= model_dicriminator
    best_acc = 0.0
    if os.path.isfile(config.svhn_trainedmodel):
        print("=> loading checkpoint '{}'".format(config.svhn_trainedmodel))
        checkpoint = torch.load(config.svhn_trainedmodel)
        trained_svhn_dicscrimator_model.load_state_dict(checkpoint['state_dict']) # fixed weight , bias for network 
    
    mnist_iter = iter(mnist_test_loader)

    svhn_iter = iter(svhn_test_loader)
    iter_per_epoch = min(len(svhn_iter), len(mnist_iter))
    if not os.path.isfile(config.svhn_trainedmodel):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('--------------------------------------------')
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model_generator.train()
                    #model.train(True)
                else:
                    model_generator.eval()
                    # model.train(False)
                       
                                                     
                running_loss =0.0
                running_corrects = 0
                
                
                
                
                #Iterate over data.
                for step in range(40000):
                    # get the inputs
                    # inputs, labels = data
                    # reset data_iter for each epoch
                    if (step+1) % iter_per_epoch == 0:
                        mnist_iter = iter(mnist_test_loader)
                        svhn_iter = iter(svhn_test_loader)
                    
                    fixed_svhn = Variable(svhn_iter.next()[0].cuda()) 
                    # print("svhn_iter.next()[0]  :", svhn_iter.next()[0])
                    # print("-------------------------------------------------------")
                    # load svhn and mnist dataset
                    svhn, s_labels = svhn_iter.next() 

                    svhn, s_labels = Variable(svhn.cuda()), Variable(s_labels.cuda()).long().squeeze()
                   # print("svhn size :", svhn.size(0))
                   # print("-------------------------------------------------------")
                    mnist, m_labels = mnist_iter.next() 
                    
                    
                    mnist_3ch = torch.FloatTensor(mnist.size(0), 3, mnist.size(2), mnist.size(3))
                    mnist_3ch[:,0,:,:].copy_(mnist)
                    mnist_3ch[:,1,:,:].copy_(mnist)
                    mnist_3ch[:,2,:,:].copy_(mnist)
                    print(" mnist_3ch size :", mnist_3ch.size())
                
                    
                    
                    mnist, m_labels = Variable(mnist.cuda()), Variable(m_labels.cuda())
                    
                    mnist_fake_labels = torch.Tensor(config.batch_size*svhn.size(0))
                    mnist_fake_labels = Variable(mnist_fake_labels.cuda()).long()
                    svhn_fake_labels = torch.Tensor(config.batch_size*mnist.size(0))
                    svhn_fake_labels = Variable(svhn_fake_labels.cuda()).long()
                    
                    
                    #######################Training D ##############################
                    # train with real images(mnist)
                    # zero the parameter gradients
                    optimizer_d1.zero_grad()
                    optimizer_d2.zero_grad()
                    optimizer_d3.zero_grad()
                    optimizer_g.zero_grad()
            
                    out = model_dicriminator(mnist)
                    print("out size: ", out.size()) # [64 ,11]
                    d3_loss = criterion(out, m_labels)
                    d_mnist_loss = d3_loss
                    
                    # train with fake images
                    
                    # forward ( LD first term )
                    fake_mnist = model_generator(svhn)
                    print('fake_mnist size : ', fake_mnist.size())
                    outputs = model_dicriminator1(fake_mnist)
                    print('outputs fake_mnist size : ', fake_mnist.size())
                    
                    d1_loss = criterion(outputs, mnist_fake_labels)
                    
                    # forward ( LD second term )
                    fake_svhn = model_generator(mnist_3ch)
                    print('fake_mnist size : ', fake_mnist.size())
                    outputs = model_dicriminator2(fake_mnist)
                    print('outputs fake_mnist size : ', fake_mnist.size())
                    
                    d2_loss = criterion(outputs, mnist_fake_labels)
                    
                    
                    
                    reconst_svhn = model_generator(fixed_svhn)
                    #print("1 reconst_svhn size : ", reconst_svhn.size())
                    reconst_svhn = reconst_svhn.cpu().data.numpy()
                    # print("2 reconst_svhn size : ", reconst_svhn.shape)
                    fixed_svhn = fixed_svhn.cpu().data.numpy()
                    
                    merged = merge_images(fixed_svhn, reconst_svhn)
                    path = os.path.join('./', 'sample-%d-m-s.png' %(step+1))
                    scipy.misc.imsave(path, merged)
                    # print("-----------------outputs------------------------")
                    # print(outputs)  # FloatTensor of size 4x10
                    # print("-----------------labels------------------------")
                    # print(labels)   # LongTensor of size 4
                    _ , preds = torch.max(outputs.data , 1) # max index with row
                    # print("-----------------prediction--------------------")
                    # print(preds)    # LongTensor of size 4 
                    
                    loss = criterion(outputs, s_labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_g.step()
                        optimizer_d.step()
                       #  print("loss" , loss)
                    # statistics
                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds == s_labels.data)
                    
                # statistics
                epoch_loss = running_loss/len_svhn_extra_train_loader   
                epoch_acc = running_corrects /len_svhn_extra_train_loader 
                
                print('{} Loss : {:.4f} Acc : {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    is_best = best_acc
                    #
                    # best_model = copy.deepcopy(model)
                    # save_checkpoint({'epoch': epoch+1 , 'arch': config.arch, 'state_dict' : model.state_dict(), 'best_acc': best_acc},is_best)
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
model_gen = model.G().cuda()
model_disc_1 = model.D1().cuda()
model_disc_2 = model.D1().cuda()
model_disc_3 = model.D1().cuda()
#print("22")


criterion = nn.CrossEntropyLoss()         
# optimzer_ft = optim.SGD(model_ft.parameters(), lr=0.0002, momentum=0.9)
# optimzer_ft = optim.Adam(model_ft.parameters(), 0.02, [0.5, 0.9999])
optimzer_g = optim.Adam(model_gen.parameters(), 0.02, [0.5, 0.9999])
optimzer_d1 = optim.Adam(model_disc.parameters(), 0.02, [0.5, 0.9999])
optimzer_d2 = optim.Adam(model_disc.parameters(), 0.02, [0.5, 0.9999])
optimzer_d3 = optim.Adam(model_disc.parameters(), 0.02, [0.5, 0.9999])
       
# Train and Evaluate
#print("33")
# model_ft = train_model(model_ft, criterion, optimzer_ft, num_epochs=5)    
# model_ft = test_model(model_ft, criterion, optimzer_ft, num_epochs=3)    

model_ft = train_generated_model_(model_gen, model_disc_1, model_disc_2,model_disc_3,criterion, optimzer_g, optimzer_d1, optimzer_d2,optimzer_d3,num_epochs=5)  
"""
for epoch in range(23):
    train_model_v2(model_ft, criterion, optimzer_ft, epoch)
    test_model_v2(model_ft, criterion, optimzer_ft, epoch)
"""    
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