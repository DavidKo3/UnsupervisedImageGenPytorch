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
    def __init__(self, conv_dim=64, use_labels=False):
        super(D1, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*2, 4)
        self.fc = nn.Linear(512 , 10) # feature 128 x [2x2]
        n_out = 11 if use_labels else 1

    def forward(self, x):
        out = F.relu(self.conv1(x))   # (?, 64, 16, 16)
        out = F.relu(self.conv2(out))   # (?, 128, 8, 8)
        out = F.relu(self.conv3(out))   # (?, 256, 4, 4)
        out = F.relu(self.conv4(out))   # (?, 128, 4, 4)
        # print("result of out ", out.size())
        # out = [4, 128, 2 ,2 ]
        # output size = (input soze + 2 x Padding - Filter size )/ Stride +1 
        (_, C, H, W) = out.data.size()
        out = out.view( -1 , C * H * W)   
        out = self.fc(out).squeeze()
    
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
    def __init__(self, conv_dim=64, use_labels=False):
        super(D2, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out
    
