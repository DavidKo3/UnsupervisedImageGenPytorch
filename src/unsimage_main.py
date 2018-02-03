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
from sqlalchemy.sql.util import criterion_as_pairs



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--arch', type=str, default='svhnDiscrimanator')
parser.add_argument('--num_iter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate, default=0.02')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--alphaCONST', type=float, default=15, help='alpha weight')
parser.add_argument('--betaCONST', type=float, default=15, help='beta weight')
parser.add_argument('--startepoch', type=int, default=0)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--svhn_path', type=str, default='./datasets/svhn')
parser.add_argument('--mnist_path', type=str , default='./datasets/mnist')
parser.add_argument('--svhn_trainedmodel', type=str, default='./svhn_extra_mnist_model_best.pth.tar')
config = parser.parse_args()
print(config)
print(config.mnist_path)

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



def train_test_model(model, criterion, optimizer, num_epochs=5):
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
            training_loss =0.0
            training_corrects = 0
            test_loss =0.0
            test_corrects = 0
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    #model.train(True)
                else:
                    model.eval()
                    # model.train(False)
                data_reloader = iter(data_loader[phase])
                #Iterate over data.
                for step , data in enumerate(data_reloader):
                    # get the inputs
                    inputs, labels = data
                    labels -= 1 
                    labels = labels.type(torch.LongTensor)       
                    labels = labels.long().squeeze() # from svhan labels byte to long and reform size 4x1 to size 4      
                    if phase == 'train':
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        _, outputs = model(inputs)
                        _ , preds = torch.max(outputs.data , 1) # max index with row
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        # statistics
                        training_loss += loss.data[0]
                        training_corrects += torch.sum(preds == labels.data)
                        avg_train_loss = training_loss/len_data_loader['train'] 
                        avg_train_corrects = training_corrects/len_data_loader['train'] 
                        print('\nepoch :{} , step:{} , average train_loss:{:.4f}, average train_corrrects:{:.4f}'.format(epoch, step, avg_train_loss, avg_train_corrects))
                    else:
                        inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda())
                        _, outputs = model(inputs)
                        _ , preds = torch.max(outputs.data , 1) # max index with row
                        loss = criterion(outputs, labels)
                        test_loss += loss.data[0]
                        test_corrects += torch.sum(preds == labels.data)
                        avg_test_loss = test_loss/len_data_loader['val'] 
                        avg_test_corrects = test_corrects/len_data_loader['val']
                        print('\nepoch :{} , step:{} ,average test_loss:{:.4f}, average test_corrects:{:.4f}'.format(epoch, step, avg_test_loss, avg_test_corrects))
                    
                print()
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
                    best_model = copy.deepcopy(model)
                    save_checkpoint({'epoch': epoch+1 , 'arch': config.arch, 'state_dict' : model.state_dict(), 'best_acc': best_acc},is_best)
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val test Acc: {:4f}'.format(best_acc))
    return best_model

def train_generated_model_2(model_generator, model_encoder, model_disc , model_dicriminator2, criterion, criterionMSE, optimizer_g, optimizer_d,num_epochs=1):
    since = time.time()
    
    # trained_svhn_dicscrimator_model= model_dicriminator
    best_acc = 0.0
    if os.path.isfile(config.svhn_trainedmodel):
        print("=> loading checkpoint '{}'".format(config.svhn_trainedmodel))
        checkpoint = torch.load(config.svhn_trainedmodel)
        model_encoder.load_state_dict(checkpoint['state_dict']) # fixed weight , bias for network 
        print('model_encoder :', model_encoder.state_dict())
   
        model_generator.train()
        model_encoder.eval()
        model_disc.train()
               
        for epoch in range(6):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('--------------------------------------------')
            
            mnist_iter = iter(mnist_train_loader)
            svhn_iter = iter(svhn_extra_train_loader)    
            iter_per_epoch = min(len(svhn_iter), len(mnist_iter))
            #Iterate over data.
            for step in range(iter_per_epoch-1 ):
                
                fixed_svhn = Variable(svhn_iter.next()[0].cuda()) 
                # load svhn and mnist dataset
                svhn, s_labels = svhn_iter.next() 
                s_labels -= 1 # svhn ranged from 1 to 10
                svhn, s_labels = Variable(svhn.cuda()), Variable(s_labels.cuda()).long().squeeze()
                # print("svhn size :", svhn.size(0))
                # print("-------------------------------------------------------")
                mnist, m_labels = mnist_iter.next() 
                    
                    
                mnist_3ch = torch.FloatTensor(mnist.size(0), 3, mnist.size(2), mnist.size(3))
                mnist_3ch[:,0,:,:].copy_(mnist)
                mnist_3ch[:,1,:,:].copy_(mnist)
                mnist_3ch[:,2,:,:].copy_(mnist)
                mnist_3ch = Variable(mnist_3ch.cuda())
                #print(" mnist_3ch size :", mnist_3ch.size())
                      
                mnist, m_labels = Variable(mnist.cuda()), Variable(m_labels.cuda())
            
                label_disc = torch.LongTensor(config.batch_size)
                label_disc = Variable(label_disc.cuda())
                label_gen = torch.LongTensor(config.batch_size)
                label_gen = Variable(label_gen.cuda())
                    
                    
                fake_source_label = 0
                fake_target_label = 1
                real_target_label = 2 
               
                # zero the parameter gradients
                optimizer_d.zero_grad()
                optimizer_g.zero_grad()
                     
                #######################Training GAND ##############################
               
                # NOTE: max_D first
                for p in model_disc.parameters(): 
                    p.requires_grad = True 
                model_disc.zero_grad()

                    
                # forward (LD last term )
                label_disc.data.resize_(config.batch_size).fill_(real_target_label)
                out_real_mnist = model_disc(mnist)
                # print("out_real_mnist : ", out_real_mnist.size())
                d_real_target_loss = criterion(out_real_mnist, label_disc)
                # print("d_real_target_loss : ", d_real_target_loss)
                d_real_target_loss.backward()
                    
                # forward ( LD second term )
                encoded_mnist, _ = model_encoder(mnist_3ch)
                faked_mnist = model_generator(encoded_mnist)
                generated_faked_mnist = faked_mnist.detach()
                outputs_faked_mnist = model_disc(generated_faked_mnist)
                label_disc.data.fill_(fake_target_label)
                d_fake_target_loss = criterion(outputs_faked_mnist, label_disc)
                d_fake_target_loss.backward()
                    
                # forward ( LD first term )
                encoded_svhn, _ = model_encoder(svhn)
                #print(' 1 fake_mnist size : ', encoded_mnist.size()) # [64, 128, 1, 1]
                faked_svhn = model_generator(encoded_svhn) # [64, 1, 32, 32]
                generated_faked_svhn = faked_svhn.detach()
                #print('2 outputs fake_mnist size : ', faked_mnist.size())
                outputs_faked_src = model_disc(generated_faked_svhn)
                    
                    
                label_disc.data.fill_(fake_source_label)
                # print('outputs_faked_src size : ', outputs_faked_src.size())
                #print('label_disc size : ', label_disc.size())
                d_faked_src_loss = criterion(outputs_faked_src, label_disc)
                #print("d_faked_src_loss :", d_faked_src_loss)
                d_faked_src_loss.backward()
                    
                # Loss D
                Loss_D = d_real_target_loss + d_fake_target_loss + d_faked_src_loss
                # update paramters to max_discriminator
                optimizer_d.step()
                    
                    
                
                # freeze computing gradients of weights in Discriminator
                for p in model_disc.parameters():
                    p.requires_grad = False
                model_generator.zero_grad()
                
                   
                # computation for LCONST
                label_gen.data.resize_(config.batch_size).copy_(s_labels.data)
                # print("label_gen :" , label_gen.size())
                # print("label_gen :" , label_gen.data)
                   
                faked_svhn_3ch = torch.FloatTensor(faked_svhn.size(0), 3, faked_svhn.size(2), faked_svhn.size(3))
                faked_svhn_3ch[:,0,:,:].copy_(faked_svhn.data)
                faked_svhn_3ch[:,1,:,:].copy_(faked_svhn.data)
                faked_svhn_3ch[:,2,:,:].copy_(faked_svhn.data)
                faked_svhn_3ch = Variable(faked_svhn_3ch.cuda())
            
                encoded_faked_svhn, _ = model_encoder(faked_svhn_3ch)
                Loss_CONST = criterionMSE(encoded_faked_svhn, encoded_svhn.detach())
                Loss_CONST = config.alphaCONST*Loss_CONST
                Loss_CONST.backward(retain_variables=True)
                  
                  
                # computation for LTID
                Loss_TID = criterionMSE(faked_mnist, mnist)
                Loss_TID = config.betaCONST*Loss_TID
                Loss_TID.backward(retain_variables=True)
                 
                    
                #######################Training GAND ##############################
                label_disc.data.resize_(config.batch_size).fill_(real_target_label)
                out_faked_src = model_disc(faked_svhn)
                loss_gan_src = criterion(out_faked_src, label_disc)
                loss_gan_src.backward()
                    
                out_faked_target = model_disc(faked_mnist)
                loss_gan_target = criterion(out_faked_target, label_disc)
                loss_gan_target.backward()
                          
                # Loss G 
                Loss_G =  loss_gan_src + loss_gan_target
                # update parameters 
                optimizer_g.step()
                
                #L_D = Loss_D.cpu().data.numpy()
                #L_G = Loss_G.cpu().data.numpy()
                #L_con = Loss_CONST.cpu().data.numpy()
                # L_tid = Loss_TID.cpu().data.numpy()
                print("\n================================================================================")   
                #print('\n  epoch :{} , step :{}, Loss_D :{} , Loss_G :{} , Loss_CONST :{} , Loss_TID :{}'.format(epoch, step, L_D, L_G, L_con, L_tid))
                print('\n epoch :{} , step :{}, Loss_D :{} , Loss_G :{} , Loss_CONST :{} , Loss_TID :{}'.format(epoch , step, Loss_D.data[0], Loss_G.data[0],Loss_CONST.data[0] ,Loss_TID.data[0]))
                print("\n================================================================================")  
                err_Loss_D, err_Loss_G = np.abs(3 -Loss_D.data[0]),  np.abs(4 -Loss_G.data[0])
                # err_Loss_D, err_Loss_G = np.abs(3 -L_D),  np.abs(4 -L_G)
                
                
                if(err_Loss_D <0.000005 and err_Loss_G <0.00005 ):
                    print("\n best Loss_D :{}, Loss_G :{}".format(Loss_D.data[0], Loss_G.data[0]))
                    # print("\n best Loss_D :{}, Loss_G :{}".format(L_D, L_G))
                
                fixed_encoded_svhn, _ = model_encoder(fixed_svhn)
              
                fixed_reconst_svhn = model_generator(fixed_encoded_svhn)
               
                fixed_reconst_svhn = fixed_reconst_svhn.cpu().data.numpy()

                fixed_svhn = fixed_svhn.cpu().data.numpy()
                    
                merged = merge_images(fixed_svhn, fixed_reconst_svhn)
                path='./results/'
                if not os.path.exists(path):
                    os.mkdir(path)
                if step % 300 ==0:
                    path = os.path.join(path, 'epoch%d-sample-%d-s-.png' %(epoch, step+1))
                    scipy.misc.imsave(path, merged)
                
                """
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    is_best = best_acc
                    #
                    # best_model = copy.deepcopy(model)
                    # save_checkpoint({'epoch': epoch+1 , 'arch': config.arch, 'state_dict' : model.state_dict(), 'best_acc': best_acc},is_best)
                """
            start=False
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    # return best_model



use_gpu = torch.cuda.is_available()
    
if torch.cuda.is_available() :
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")     
         
#print("11")
model_gen = model.G1().cuda()
model_encoder = model.E().cuda() # f:= feature extractor
model_disc = model.D1().cuda()
model_disc_2 = model.D_SVHN().cuda()


criterion = nn.CrossEntropyLoss()     
criterionMSE = nn.MSELoss()  

optimzer_f = optim.Adam(model_encoder.parameters(), 0.02, betas= (config.beta1, 0.999))
optimzer_g = optim.Adam(model_gen.parameters(), 0.02, betas= (config.beta1, 0.999))
optimzer_disc = optim.Adam(model_disc.parameters(), 0.02, betas= (config.beta1, 0.999))
 
 
# training step for Feature extractor 
# model_encoder = train_test_model(model_encoder, criterion,optimzer_f)
  
model_ft = train_generated_model_2(model_gen, model_encoder, model_disc,model_disc_2 ,criterion, criterionMSE, optimzer_g, optimzer_disc,num_epochs=1)
     
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