import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch



def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    
    layers=[]
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

# generate input sample and forward to get shape
def _get_conv_output(shape):
    bs = 1
    input = Variable(torch.rand(bs, *shape))
    output_feat = self._forward_features(input)
    n_size = output_feat.data.view(bs, -1).size(1)
    return n_size
    

class D1(nn.Module):
    """Discriminator for mnist."""
    def __init__(self, conv_dim=64, use_labels=True):
        super(D1, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)     
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*2, 6)
        # self.fc = nn.Linear(conv_dim*2 , 10) # feature size 128 x [2x2]
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*2, n_out, 1, 1, 0, False)

    def forward(self, x):
        out = F.relu(self.conv1(x))   # (?, 64, 16, 16)    , (32 + 2x1 - 4)/2+1 = 16
        out = F.relu(self.conv2(out))   # (?, 128, 8, 8)
        out = F.relu(self.conv3(out))   # (?, 256, 4, 4)
        out = F.relu(self.conv4(out))   # (?, 128, 1, 1)
   
       # print("result of out :", out.size())
        # out = [4, 128, 2 ,2 ]
        # output size = (input soze + 2 x Padding - Filter size )/ Stride +1 
       # (_, C, H, W) = out.data.size()
       # print("before view out size :" , out.size())
       # out = out.view( -1 , C * H * W)   
       # print("after view out size :" , out.size())
       # print("===================before squeeze out===========================")
       # print(out) # [4, 128 ,2 , 2]
        #print("===================before squeeze and fc(out) ===========================")
        #print(self.fc(out).size()) # [4, 512]
        out = self.fc(out).squeeze()
        #print("===================before squeeze out===========================")
        #print(out) # [4, 512]

        return out

    
    
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
    
class D2(nn.Module):
    """Discriminator for svhn."""
    def __init__(self, conv_dim=64, use_labels=True):
        super(D2, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*2, 6)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*2, n_out, 1, 1, 0, False)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))   # (?, 64, 16, 16)
        #print("out1 ", out)
        out = F.relu(self.conv2(out))   # (?, 128, 8, 8)
        #print("out1 ", out)
        out = F.relu(self.conv3(out))   # (?, 256, 4, 4)
        # print("out1 ", out)
        out = F.relu(self.conv4(out))   # (?, 128, 1, 1)
        # print("result of out ", out.size())
        # out = [4, 128, 2 ,2 ]
        # output size = (input soze + 2 x Padding - Filter size )/ Stride +1 
       # (_, C, H, W) = out.data.size()
       # print("before view out size :" , out.size())
       # out = out.view( -1 , C * H * W)   
       # print("after view out size :" , out.size())
       # print("===================before squeeze out===========================")
       # print(out) # [4, 128 ,2 , 2]
       # print("===================before squeeze and fc(out) ===========================")
       # print(self.fc(out)) # [4, 11]
        out = self.fc(out).squeeze()
       # print("===================before squeeze out===========================")
        # print(out.size()) # [4, 11]
        return out

    
    
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
    
class G(nn.Module):
    """Generator for transfering svhn to mnist """
    def __init__(self, conv_dim=64, use_labels=True):
        super(G, self).__init__()
        # Encoding 
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        
         # residual blocks
        self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        
        # Decoding 
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 1,4, bn=False)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))   # (?, 64, 16, 16)
        #print('conv1 ', out.size())
        out = F.relu(self.conv2(out))   # (?, 128, 8, 8)
        # print('conv2 ', out.size())
        out = F.relu(self.deconv1(out))   # (?, 64, 16, 16)
        # print('deconv1 ', out.size())
        out = F.tanh(self.deconv2(out))   # (?, 1, 32, 32)
        # print("deconv2", out.size())
        return out

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