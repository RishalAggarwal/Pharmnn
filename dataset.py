import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.optim.lr_scheduler import StepLR
import sklearn.metrics
import wandb
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import argparse
import multiprocessing, time
import molgrid
import pickle
from gridData import Grid
from random import sample
try:
    from molgrid.openbabel import pybel
except ImportError:
    from openbabel import pybel

import random
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
    def __init__(self, txt_file, feat_to_int,int_to_feat,top_dir='.',grid_dimension=5,use_gist=True,resolution=0.5, rotate=True,autobox_extend=4,cache=None,classcnts=None,coordcache=None):
        super(PharmacophoreDataset, self).__init__()
        self.int_to_feat=int_to_feat
        self.use_gist = use_gist
        self.rotate = rotate
        self.autobox_extend=autobox_extend
        self.resolution=resolution
        self.gmaker = MyGridMaker(resolution=resolution, dimension=grid_dimension) 
        self.dims = self.gmaker.g.grid_dimensions(molgrid.defaultGninaReceptorTyper.num_types())
        self.cache=None
        s = molgrid.ExampleProviderSettings(data_root=top_dir)
        coord_reader = molgrid.CoordCache(molgrid.defaultGninaReceptorTyper,s) 
        #preloaded dataset
        if txt_file.endswith('.pkl'):
            self.cache=cache
            self.classcnts=classcnts
            self.coordcache=coordcache
        else:
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
            sdf_paths = np.asarray(data_info.iloc[:, -2])
            N = len(labels)
            self.classcnts = np.zeros(len(feat_to_int))
            
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

            self.cache = []
            self.coordcache = dict()
            for lnames, fcoord, gcoord, pdbfile,sdffile, gistcenter in zip(labels, centers, grid_centers, pdb_paths,sdf_paths, gists):
                gist,gcenter = gistcenter
                feat_label = np.zeros(len(feat_to_int))
                for lname in lnames.split(':'):
                    if lname in feat_to_int:
                        feat_label[feat_to_int[lname]] = 1.0
                if gist.size == 0: #don't have gist grids, use feature as center
                    gcenter = tuple(fcoord)
                if pdbfile not in self.coordcache:
                    self.coordcache[pdbfile] = MyCoordinateSet(coord_reader.make_coords(pdbfile))
                if sdffile not in self.coordcache:
                    self.coordcache[sdffile] = MyCoordinateSet(coord_reader.make_coords(sdffile))
                self.cache.append({'label': feat_label,
                                    'fcoord': fcoord,
                                    'gcoord': gcoord,
                                    'gcenter': gcenter,
                                    'pdbfile': pdbfile,
                                    'sdffile': sdffile,
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
        mask=torch.ones(len(self.int_to_feat))
        coords.togpu(True)
        self.grid_protein(pdb_grid,coords,example['gcenter'])
        if self.use_gist:
            return {'label': example['label'],
                    'grid': torch.concat([example['gist'],pdb_grid],axis=0)
                    }
        else:
            return {'label': torch.tensor(example['label']),
                    'grid': pdb_grid,
                    'mask': mask,
                    'center': example['gcenter'],
                    'pdbfile': example['pdbfile'],
                    'sdffile': example['sdffile']
                    }

    def __len__(self):
        return len(self.cache)
    
    def grid_protein(self,pdb_grid,coords,gcenter):
        if self.rotate:
            t = molgrid.Transform(gcenter,random_rotation=True)
            t.forward(coords,coords)
        self.gmaker.g.forward(gcenter, coords, pdb_grid)        
    
    

    def binding_site_grids(self,pdbfile,sdffile):
        ligand_coords = self.coordcache[sdffile].c.clone().coords.tonumpy()
        autobox_coords=autobox_ligand(ligand_coords,self.autobox_extend)
        coords = self.coordcache[pdbfile].c.clone()
        coords.togpu(True)
        for center in autobox_coords:
            gcenter=tuple(center)
            pdb_grid = torch.zeros(self.dims,dtype=torch.float32,device="cuda")
            grid_protein(pdb_grid,coords,gcenter,self.gmaker,self.rotate)
            yield center,pdb_grid
    
    def get_complexes(self):
        seen_before=[]
        for example in self.cache:
            if [example['sdffile'],example['pdbfile']] in seen_before:
                continue
            seen_before.append([example['sdffile'],example['pdbfile']])
        return seen_before


class NegativesDataset(Dataset):
    def __init__(self,negatives_text_file,pharm_dataset,dataset_size=1e7):
        self.pharm_dataset=pharm_dataset
        self.classcnts=self.pharm_dataset.classcnts
        self.txt_file=negatives_text_file
        if self.txt_file.endswith('.pkl'):
            self.cache = pickle.load(open(self.txt_file,'rb'))
        else:
            # Subsample large dataset file
            # By Massoud Seifi - http://metadatascience.com/2014/02/27/random-sampling-from-very-large-files/
            self.cache={}
            with open(self.txt_file, 'r') as f:
                f.seek(0, 2)
                filesize = f.tell()
                random_set = np.sort(np.random.randint(1, filesize, size=int(dataset_size)))
                for i in random_set:
                    f.seek(i)
                    # Skip current line (because we might be in the middle of a line) 
                    f.readline()
                    # Append the next line to the sample set 
                    new_line=f.readline().rstrip()
                    tokens=new_line.split(',')
                    try:
                        if tokens[0].split('Not')[1] in self.cache.keys():
                            self.cache[tokens[0].split('Not')[1]].append({
                            'pdbfile': tokens[-1],
                            'sdffile': tokens[-2],
                            'center':tuple(map(float,tokens[1:4]))})
                        else:
                            self.cache[tokens[0].split('Not')[1]]=[{
                            'pdbfile': tokens[-1],
                            'sdffile': tokens[-2],
                            'center':tuple(map(float,tokens[1:4]))}]  
                    except:
                        continue
    
    def __len__(self):
        return self.pharm_dataset.__len__()
    
    def __getitem__(self, index):
        pharm_data_item=self.pharm_dataset.__getitem__(index)
        labels=np.argwhere(pharm_data_item['label'].numpy()==1)
        pharm_data_item['label']=pharm_data_item['label'].unsqueeze(0)
        pharm_data_item['grid']=pharm_data_item['grid'].unsqueeze(0)
        pharm_data_item['mask']=pharm_data_item['mask'].unsqueeze(0)
        label=sample(list(labels),1)[0]
        label_arr=torch.zeros(len(self.pharm_dataset.int_to_feat))
        mask=torch.zeros((len(self.pharm_dataset.int_to_feat)))
        mask[label]=1.0
        label_feat=self.pharm_dataset.int_to_feat[label[0]]
        negative_point=sample(self.cache[label_feat],1)[0]
        pdb_grid = torch.zeros(self.pharm_dataset.dims,dtype=torch.float32,device="cuda")
        coords = self.pharm_dataset.coordcache[negative_point['pdbfile']].c.clone()
        coords.togpu(True)
        self.pharm_dataset.grid_protein(pdb_grid,coords,negative_point['center'])
        pharm_data_item['label']=torch.concat([pharm_data_item['label'],label_arr.unsqueeze(0)],axis=0)   
        pharm_data_item['grid']=torch.concat([pharm_data_item['grid'],pdb_grid.unsqueeze(0)],axis=0)
        pharm_data_item['mask']=torch.concat([pharm_data_item['mask'],mask.unsqueeze(0)],axis=0)
        return pharm_data_item 
    

class Inference_Dataset(Dataset):

    def __init__(self,receptor,ligand,feature_points=None,auto_box_extend=4,grid_dimension=5,resolution=0.5,rotate=False):
        super(Inference_Dataset, self).__init__()
        self.receptor=receptor
        self.ligand=ligand
        self.receptor_coords=MyCoordinateSet(self.receptor)
        self.ligand_coords=MyCoordinateSet(self.ligand)
        self.auto_box_extend=auto_box_extend
        self.gmaker=MyGridMaker(resolution=resolution, dimension=grid_dimension)
        self.dims = self.gmaker.g.grid_dimensions(molgrid.defaultGninaReceptorTyper.num_types())
        self.rotate=rotate
        self.points=None
        self.resolution=resolution

    def __getitem__(self, index):
        pdb_grid = torch.zeros(self.dims,dtype=torch.float32,device="cuda")
        coords = self.receptor_coords.c.clone()
        coords.togpu(True)
        center=self.points.loc[index][1:4]
        center=tuple(center.tolist())
        grid_protein(pdb_grid,coords,center,self.gmaker,self.rotate)
        return {'grid': pdb_grid}

    def __len__(self):
        return len(self.points)

    def get_complexes(self):
        return [[self.ligand_coords,self.receptor_coords]]
    
    def binding_site_grids(self,receptor,ligand):
        ligand_coords = ligand.c.clone().coords.tonumpy()
        autobox_coords=autobox_ligand(ligand_coords,self.auto_box_extend)
        coords = receptor.c.clone()
        coords.togpu(True)
        for center in autobox_coords:
            gcenter=tuple(center)
            pdb_grid = torch.zeros(self.dims,dtype=torch.float32,device="cuda")
            grid_protein(pdb_grid,coords,gcenter,self.gmaker,self.rotate)
            yield center,pdb_grid

    def add_points(self,points):
        if self.points is None:
            self.points=points
        #concatenate rows of new dataframe (except for first row which is the label)
        else:
            self.points.append(points,ignore_index=True)
        self.points.reset_index(drop=True,inplace=True)
        #TODO cluster points of same class
    
    def get_points(self):
        return np.array(self.points)


            
        
        




def autobox_ligand(coords,autobox_extend=4):    
    max_x=np.max(coords[:,0])
    min_x=np.min(coords[:,0])
    
    max_y=np.max(coords[:,1])
    min_y=np.min(coords[:,1])
    
    max_z=np.max(coords[:,2])
    min_z=np.min(coords[:,2])
    
    num_x=int((max_x-min_x+2*autobox_extend)/0.5)
    num_y=int((max_y-min_y+2*autobox_extend)/0.5)
    num_z=int((max_z-min_z+2*autobox_extend)/0.5)
    
    coords_x=np.linspace(min_x-autobox_extend,min_x-autobox_extend+(num_x/2),num_x+1)
    coords_y=np.linspace(min_y-autobox_extend,min_y-autobox_extend+(num_y/2),num_y+1)
    coords_z=np.linspace(min_z-autobox_extend,min_z-autobox_extend+(num_z/2),num_z+1)
    
    coords_x=torch.tensor(coords_x)
    coords_y=torch.tensor(coords_y)
    coords_z=torch.tensor(coords_z)
    autobox_coords=torch.cartesian_prod(coords_x,coords_y,coords_z).numpy()
    return autobox_coords

def grid_protein(pdb_grid,coords,gcenter,gmaker,rotate=True):
    if rotate:
        t = molgrid.Transform(gcenter,random_rotation=True)
        t.forward(coords,coords)
    gmaker.g.forward(gcenter, coords, pdb_grid) 