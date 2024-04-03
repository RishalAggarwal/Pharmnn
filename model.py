import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import argparse
import molgrid
import pickle
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
        #TODO remove this dependancy 

        self.fc1 = nn.Linear(lastgd*lastgd*lastgd*w*m4, w)
        self.fc2 = nn.Linear(w, 6)
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x):
        batch_size = x.shape[0]

        #print(x.shape)
        x = self.seq(x)
        #print(x.shape)
        x = x.view(batch_size, -1)
        x1 = self.sigma(self.fc1(x))
        #print(x.shape)
        x = self.dropout(x1)
        x = self.fc2(x)
        #print(x.shape)
        return x,x1
