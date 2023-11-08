import torch
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
import numpy as np


def grid_protein(rotate,gmaker,pdb_grid,coords,gcenter):
    if rotate:
        t = molgrid.Transform(gcenter,random_rotation=False)
        t.forward(coords,coords)
    gmaker.g.forward(gcenter, coords, pdb_grid)        
    
def autobox_ligand(coordcache,autobox_extend):
    coords = coordcache.c.clone().coords.tonumpy()
    
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

def binding_site_grids(protein,ligand,rotate,gmaker):
    autobox_coords=autobox_ligand(ligand)
    coords = protein.c.clone()
    coords.togpu(True)
    for center in autobox_coords:
        gcenter=tuple(center)
        pdb_grid = torch.zeros(self.dims,dtype=torch.float32,device="cuda")
        grid_protein(rotate,gmaker,pdb_grid,coords,gcenter)
        yield center,pdb_grid