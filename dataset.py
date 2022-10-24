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
import multiprocessing, time
import molgrid
try:
    from molgrid.openbabel import pybel
except ImportError:
    from openbabel import pybel

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