#!/usr/bin/env python3

import torch
from gridData import Grid
import numpy as np
from numpy import argmax
from numpy import array
import sys,os
import pandas as pd
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
import multiprocessing, time
import pickle
import se3cnn
from se3cnn.image.convolution import SE3Convolution
from se3cnn.image.gated_block import GatedBlock

import molgrid
try:
    from molgrid.openbabel import pybel
except ImportError:
    from openbabel import pybel

pybel.ob.obErrorLog.StopLogging() #without this wandb will deadlock when ob fills up the write buffer

parser = argparse.ArgumentParser('Train a CNN on GIST data to predict pharmacophore feature.')
parser.add_argument('--train_data',required=True,help='data to train with',default="data_train_pdb.txt")
parser.add_argument('--test_data',default="",help='data to test with')
parser.add_argument('--pickle_only',help="Create pickle files of the data only; don't train",action='store_true')
parser.add_argument('--batch_size',default=256,type=int,help='batch size')
parser.add_argument('--epochs',default=265,type=int,help='number epochs')
parser.add_argument('--steplr',default=150,type=int,help='when to step the learning rate')

parser.add_argument('--lr',default=0,type=float,help='learning rate')
parser.add_argument('--solver',default='adam',help='solver to use (sgd|adam)')
parser.add_argument('--clip',default=1.0,type=float,help='gradient clipping value')
parser.add_argument('--weight_decay',default=0.0,type=float,help='weight decay')
parser.add_argument('--dropout',default=0.0,type=float,help='dropout percentage')
parser.add_argument('--conv_res',default=32,type=int,help='convolution layer resolution')
parser.add_argument('--kernel_size',default=3,type=int,help='convolution kernal size')

parser.add_argument('--block_depth',default=2,type=int,help='depth of each convolution block')
parser.add_argument('--activation',default='relu',type=str,help='pytorch name of activation function')
parser.add_argument('--expand_width',default=0,type=int,help='increase width of convolutions in each block')
parser.add_argument('--grid_dimension',default=9.5,type=float,help='dimension in angstroms of grid; only 5 is supported with gist')

parser.add_argument('--use_gist', type=int,default=0,help='use gist grids')
parser.add_argument('--rotate', type=int,default=1,help='random rotations of pdb grid')
parser.add_argument('--use_se3', type=int,default=0,help='use se3 convolutions')
parser.add_argument('--seed',default=42,type=int,help='random seed')

args = parser.parse_args()

#non-se3 conv and se3 prefer different learning rates
if args.lr == 0:
  args.lr = 0.01 if args.use_se3 else 0.001

if args.rotate and args.use_gist:
    print("Cannot enable rotation and GIST at same time yet")
    sys.exit(-1)
    
if args.grid_dimension != 5 and args.use_gist:
    print("I haven't bothered to update gist to support dimensions besides 5A")
    sys.exit(-1)
    
torch.manual_seed(args.seed)
molgrid.set_random_seed(args.seed)

wandb.init(project="pharmnn", config=args)

train_data = args.train_data
test_data = args.test_data
if not test_data: # infer test file name from train file name - makes wandb sweep easier
    test_data = train_data.replace('train','test')

#one-hot encoder vectors
category = ["Aromatic", "HydrogenAcceptor", "HydrogenDonor", "Hydrophobic", "NegativeIon", "PositiveIon"]
feat_to_int = dict((c, i) for i, c in enumerate(category))
int_to_feat = dict((i, c) for i, c in enumerate(category))

#add pickle support to CoordinateSet
class MyCoordinateSet:
    
    def __init__(self, c):
        self.c = c
        
    def __getstate__(self):
        return self.c.coords.tonumpy(),self.c.type_index.tonumpy(), self.c.radii.tonumpy(), self.c.max_type,self.c.src
        
    def __setstate__(self,vals):    
        self.c = molgrid.CoordinateSet(vals[0],vals[1],vals[2],vals[3])
    
#and gridmaker
class MyGridMaker:
    
    def __init__(self, resolution, dimension):
        self.g = molgrid.GridMaker(resolution=resolution, dimension=dimension)
        
    def __getstate__(self):
        return self.g.get_resolution(), self.g.get_dimension()
        
    def __setstate__(self,vals):    
        self.g = molgrid.GridMaker(resolution=vals[0],dimension=vals[1])
        
#Dataset class
class PharmacophoreDataset(Dataset):
    def __init__(self, txt_file, top_dir='.',grid_dimension=5,use_gist=True, rotate=True):
        super(PharmacophoreDataset, self).__init__()
        self.use_gist = use_gist
        self.rotate = rotate
        data_info = pd.read_csv(txt_file, header=None)
        self.top_dir = top_dir
        labels = np.asarray(data_info.iloc[:, 0])
        centers = np.asarray(data_info.iloc[:, 1:4])
        if data_info.shape[1] > 20:        
            grid_centers = np.asarray(data_info.iloc[:, 4:7])
            dx_paths = np.asarray(data_info.iloc[:, 7:20])
        else:
            grid_centers = np.zeros_like(centers)
            dx_paths = [[]] * data_info.shape[0]

        pdb_paths = np.asarray(data_info.iloc[:, -1])
        N = len(labels)
        self.cache = []
        self.classcnts = np.zeros(6)
        
        s = molgrid.ExampleProviderSettings(data_root=top_dir)
        coord_reader = molgrid.CoordCache(molgrid.defaultGninaReceptorTyper,s) 
        
        self.gmaker = MyGridMaker(resolution=0.5, dimension=grid_dimension) 
        self.dims = self.gmaker.g.grid_dimensions(molgrid.defaultGninaReceptorTyper.num_types())
        
        for index,lnames in enumerate(labels):
            for lname in lnames.split(':'):
                if lname in feat_to_int:
                    self.classcnts[feat_to_int[lname]] += 1.0
            
        #per example weights for weighted sampler
        c_weight = [N/(ccnt) for ccnt in self.classcnts]
        self.weights = np.zeros(N)
        for index,lnames in enumerate(labels):
            for lname in lnames.split(':'):
                if lname in feat_to_int:
                    self.weights[index] += c_weight[feat_to_int[lname]]
        
        #load gist grids in parallel
        pool = multiprocessing.Pool()
        gists = pool.map(PharmacophoreDataset.load_dx, zip(grid_centers, dx_paths))
        pool.close()

        self.coordcache = dict()
        for lnames, fcoord, gcoord, pdbfile, gistcenter in zip(labels, centers, grid_centers, pdb_paths, gists):
            gist,gcenter = gistcenter
            feat_label = np.zeros(6)
            for lname in lnames.split(':'):
                if lname in feat_to_int:
                    feat_label[feat_to_int[lname]] = 1.0
            if gist.size == 0: #don't have gist grids, use feature as center
                gcenter = tuple(fcoord)
            if pdbfile not in self.coordcache:
                self.coordcache[pdbfile] = MyCoordinateSet(coord_reader.make_coords(pdbfile))
            self.cache.append({'label': feat_label,
                                'fcoord': fcoord,
                                'gcoord': gcoord,
                                'gcenter': gcenter,
                                'pdbfile': pdbfile,
                                'gist': torch.FloatTensor(gist)})
        print("data loaded")
    
        
    @staticmethod
    def load_dx(x):
        gridCoord,dx_paths = x
        # grid coordinates for GIST
        i = int(gridCoord[0])
        j = int(gridCoord[1])
        k = int(gridCoord[2])
                
        gist_grids = []
        
        gcenter = (0,0,0)
        # GIST DATA
        for dx_file in dx_paths:
            g = Grid(dx_file)
            subgrid = g.grid[i-5:i+6, j-5:j+6, k-5:k+6]
            gist_grids.append(subgrid)        
            gcenter = g.origin + g.delta*(i,j,k) # realign centers

        gist_tensor = np.array(gist_grids,dtype=np.float32)
        gist_tensor[gist_tensor > 10] = 10
        
        return gist_tensor, tuple(gcenter)            
        
    def __getitem__(self, index):
        example = self.cache[index]
        #create pdb grid on the fly
        pdb_grid = torch.zeros(self.dims,dtype=torch.float32,device="cuda")
        coords = self.coordcache[example['pdbfile']].c.clone()
        coords.togpu(True)
        if self.rotate:
            t = molgrid.Transform(example['gcenter'],random_rotation=True)
            t.forward(coords,coords)

        self.gmaker.g.forward(example['gcenter'], coords, pdb_grid)
        if self.use_gist:
            return {'label': example['label'],
                    'grid': torch.concat([example['gist'],pdb_grid],axis=0)
                    }
        else:
            return {'label': torch.tensor(example['label']),
                    'grid': pdb_grid
                    }

    def __len__(self):
        return len(self.cache)

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


def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)

def log_metrics(prefix, labels, predicts,epoch):
    '''Given true labels and unrounded predictions calculate and log metrics.
    These should be lists of 6-vectors'''
    labels = np.array(labels)
    predicts = np.array(predicts)
    
    metrics = {'epoch':epoch}
    f1_total=0
    for cname in category:
        i = feat_to_int[cname]
        L = labels[:,i]
        P = predicts[:,i]
        # original method for creating imbalanced dataset is not viable (each protein has specific labels)
        # will make pass for now
        try:
            metrics[prefix+' '+cname+' Accuracy'] = sklearn.metrics.accuracy_score(L, np.round(P).astype(int))
            metrics[prefix+' '+cname+' Precision'] = sklearn.metrics.precision_score(L, np.round(P).astype(int))
            metrics[prefix+' '+cname+' Recall'] = sklearn.metrics.recall_score(L, np.round(P).astype(int))
            metrics[prefix+' '+cname+' F1'] = sklearn.metrics.f1_score(L, np.round(P).astype(int))
            if 'Test' in prefix:
                f1_total+=metrics[prefix+' '+cname+' F1']
            metrics[prefix+' '+cname+' AUC'] = sklearn.metrics.roc_auc_score(L, P)
        except ValueError:
            pass
    metrics['Total Test F1'] = f1_total

    print(metrics)
    wandb.log(metrics)
    
def get_dataset(fname, args):
    '''Create a dataset.  If a pkl file is not passed, create one for faster loading later'''
    if fname.endswith('.pkl'):
        dataset = pickle.load(open(fname,'rb'))
        dataset.rotate = args.rotate
        #TODO: factor out parameters that shouldn't be pickled
        dataset.use_gist = args.use_gist
        dataset.gmaker = MyGridMaker(resolution=0.5, dimension=args.grid_dimension) 
        dataset.dims = dataset.gmaker.g.grid_dimensions(molgrid.defaultGninaReceptorTyper.num_types())        
        return dataset
    else:
        dataset = PharmacophoreDataset(txt_file=fname, grid_dimension=args.grid_dimension, rotate=args.rotate, use_gist=args.use_gist)
        prefix,ext = os.path.splitext(fname)
        pickle.dump(dataset, open(prefix+'.pkl','wb'))
        return dataset
    
#Creation of test set/loader (individual system)


dataset1 = get_dataset(train_data,args)
trainloader = DataLoader(dataset1, batch_size=args.batch_size, num_workers=0, shuffle=True,drop_last=False)

dataset2 = get_dataset(test_data,args)
testloader = DataLoader(dataset2, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)


if args.pickle_only:
    sys.exit(0)

#Training
net = GISTNet(args)
print(net)
net.apply(weights_init)
net.to('cuda')
wandb.watch(net)

paramcnt = sum([np.prod(p.size()) for p in  filter(lambda p: p.requires_grad, net.parameters())])
wandb.log({'Parameters': paramcnt})
print("Parameters",paramcnt)

#calculate weights of classes
pos_weight = [(len(dataset1)-ccnt)/ccnt for ccnt in dataset1.classcnts] 
criterion = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.tensor(pos_weight).to('cuda'))

if args.solver == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
else:
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

clip_value = args.clip

change_lr = StepLR(optimizer, step_size = 1, gamma=0.1)
steplr = args.steplr

print("starting training")

for epoch in range(args.epochs):
    running_loss = 0.0
    testloss = 0.
    labels = []
    predicted = []
    
    start = time.time()
    net.train()
    for i, data in enumerate(trainloader):
        optimizer.zero_grad()
        inputs = data['grid']
        outputs = net(inputs.to('cuda'))
        loss = criterion(outputs, data['label'].to('cuda'))
        loss.backward()
        
        wandb.log({'Training Loss': loss})
        if i % 100 == 0:
            print('Training Loss', epoch, i, loss.item())
        if clip_value > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
        optimizer.step()

        
        sg_outputs = torch.sigmoid(outputs.detach().cpu()) # push through a sigmoid layer just for accuracy calculations
        predicted += sg_outputs.tolist()
        labels += data['label'].cpu().tolist()
   
    wandb.log({'Epoch':epoch})
    log_metrics('Train',labels,predicted,epoch)
    print(f"Epoch {epoch} time {time.time()-start}")
    print("started testing")
    test_predict = []
    test_labels = []
    with torch.no_grad():    
        net.eval()
        for(i, data) in enumerate(testloader):
            inputs = data['grid']
            outputs = net(inputs.to('cuda'))
            testloss = criterion(outputs, data['label'].to('cuda'))                                    
            sg_outputs = torch.sigmoid(outputs.detach().cpu())
            test_predict += sg_outputs.tolist()
            test_labels += data['label'].cpu().tolist()
    print('Learning rate is ', change_lr.get_last_lr())
    
    log_metrics('Test',test_labels,test_predict,epoch)
    wandb.log({'Learning Rate': change_lr.get_last_lr()[-1]})

    if epoch != 0 and (epoch % steplr) == 0:
        change_lr.step()
        steplr = steplr//2  #each step has twice as few iterations
    
print("finished training")
torch.save(net, os.path.join(wandb.run.dir, "model.pkl"))
