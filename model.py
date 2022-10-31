import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.optim.lr_scheduler import StepLR
import sklearn.metrics
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import argparse
import molgrid
import pickle
import se3cnn
from se3cnn.image.convolution import SE3Convolution
from se3cnn.image.gated_block import GatedBlock
try:
    from molgrid.openbabel import pybel
except ImportError:
    from openbabel import pybel

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)

#CNN class
class GISTNet(nn.Module):
    def __init__(self, args):
        super(GISTNet, self).__init__()
        nfilters = 14
        if args.use_gist:
            nfilters += 13
        
        self.sigma = getattr(F,args.activation)            
        w = args.conv_res
        ksize = args.kernel_size
        pad = ksize//2
        
        if args.expand_width:
            m2 = 2
            m4 = 4
        else:
            m2 = m4 = 1
        gd = int(args.grid_dimension*2)+1
        initk = 4
        if gd > 11:
            initk = 5
            
        penultimate_gd = ((gd-initk+1)//2)//2
        lastk = min(penultimate_gd,5)
        lastgd = (penultimate_gd-lastk+1)
        
        if args.use_se3:                        
            
            def se3tuple(width,dim):
                return tuple([width for i in range(dim+1)])
                
            blocks = []
            d = args.use_se3
            if d < 0: #hack: negative is zero
                d = 0
            stride = 1
            if args.block_depth == 1:
              stride = 2
            blocks.append(GatedBlock((nfilters,), se3tuple(w,d),size=initk,stride=stride,activation=(self.sigma,torch.sigmoid)))
            for _ in range(1,args.block_depth):
                if _ == args.block_depth-1:
                  stride = 2
                blocks.append(GatedBlock(se3tuple(w,d),se3tuple(w,d),size=ksize,padding=pad,stride=stride,activation=(self.sigma,torch.sigmoid)))
                       
            stride = 1
            if args.block_depth == 1:
              stride = 2
            blocks.append(GatedBlock(se3tuple(w,d),se3tuple(w*m2,d),size=ksize,padding=pad,stride=stride,activation=(self.sigma,torch.sigmoid)))
            for _ in range(1,args.block_depth):
                if _ == args.block_depth-1:
                  stride = 2              
                blocks.append(GatedBlock(se3tuple(w*m2,d),se3tuple(w*m2,d),size=ksize,padding=pad,stride=stride,activation=(self.sigma,torch.sigmoid)))
                
            blocks.append(GatedBlock(se3tuple(w*m2,d),(w*m4,),size=lastk,padding=0,activation=(self.sigma,torch.sigmoid)))    
            self.seq = torch.nn.Sequential(*blocks)       
            
        else:
            blocks = []
            pool = nn.MaxPool3d(2, 2) 

            blocks.append(nn.Conv3d(nfilters, w, initk,padding=0)) # 11->8
            for _ in range(1,args.block_depth):
                blocks.append(nn.Conv3d(w, w, ksize,padding=pad)) # 8->8
                
            blocks.append(pool)
            
            blocks.append(nn.Conv3d(w, w*m2, ksize,padding=pad)) # after pool this is 4
            for _ in range(1,args.block_depth):
                blocks.append(nn.Conv3d(w*m2, w*m2, ksize,padding=pad))
            
            blocks.append(pool)
            # after pool we are at 2, not much point in having multiple convs
            blocks.append(nn.Conv3d(w*m2, w*m4, lastk,padding=0))
            
            self.seq = torch.nn.Sequential(*blocks)

        self.fc1 = nn.Linear(lastgd*w*m4, w)
        self.fc2 = nn.Linear(w, 6)
        
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x):
        batch_size = x.shape[0]

        #print(x.shape)
        x = self.seq(x)
        #print(x.shape)
        x = x.view(batch_size, -1)
        x = self.sigma(self.fc1(x))
        #print(x.shape)
        x = self.dropout(x)
        x = self.fc2(x)
        #print(x.shape)
        return x
