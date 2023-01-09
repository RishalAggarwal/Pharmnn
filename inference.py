import torch
import numpy as np
from numpy import argmax
from numpy import array
import sys,os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.autograd import Variable
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
from model import GISTNet, weights_init
from dataset import MyCoordinateSet, MyGridMaker,PharmacophoreDataset
from train_pharmnn import get_dataset
from scipy.spatial import distance_matrix
from skimage.measure import label
from sklearn.cluster import AgglomerativeClustering
import molgrid
try:
    from molgrid.openbabel import pybel
except ImportError:
    from openbabel import pybel

pybel.ob.obErrorLog.StopLogging() #without this wandb will deadlock when ob fills up the write buffer

def parse_arguments():
    parser = argparse.ArgumentParser('Use a CNN on GIST data to predict pharmacophore feature.')
    parser.add_argument('--train_data',help='data to train with',default="")
    parser.add_argument('--test_data',default="",help='data to test with')
    parser.add_argument('--create_dataset',action='store_true',help='create new dataset with verified negatives')
    parser.add_argument('--top_dir',default='./data/',help='root directory of data')
    parser.add_argument('--batch_size',default=256,type=int,help='batch size')
    parser.add_argument('--expand_width',default=0,type=int,help='increase width of convolutions in each block')
    parser.add_argument('--grid_dimension',default=9.5,type=float,help='dimension in angstroms of grid; only 5 is supported with gist')
    parser.add_argument('--dataset_threshold',default=0.9,type=float,help='greedy threshold at which to sample points for active learning')
    parser.add_argument('--use_gist', type=int,default=0,help='use gist grids')
    parser.add_argument('--rotate', type=int,default=0,help='random rotations of pdb grid')
    parser.add_argument('--use_se3', type=int,default=0,help='use se3 convolutions')
    parser.add_argument('--seed',default=42,type=int,help='random seed')
    parser.add_argument('--autobox_extend',default=4,type=int,help='amount to expand autobox by')
    parser.add_argument('--create_dx',help="Create dx files of the predictions",action='store_true')
    parser.add_argument('--output_pred',help="output predictions into a text file",action='store_true')
    parser.add_argument('--round_pred',help="round up predictions in dx files",action='store_true')
    parser.add_argument('--model',type=str,help='model to test with')
    parser.add_argument('--output',type=str,help='output file of the predictions')
    parser.add_argument('--negative_output',type=str,help='output file of negative predictions')
    parser.add_argument('--prefix_dx',type=str,default="",help='prefix for dx files')
    parser.add_argument('--prefix_xyz',type=str,default="",help='prefix for xyz files')
    parser.add_argument('--verbose',help="verbse complex working on",action='store_true')
    parser.add_argument('--spherical',help="spherical masking of generated densities",action='store_true')
    parser.add_argument('--xyz',help="output xyz files of pharmacophoress",action='store_true')
    parser.add_argument('--prob_threshold',default=0.9,type=float,help='probability threshold for masking')
    parser.add_argument('--clus_threshold',default=1.5,type=float,help='distance threshold for clustering pharmacophore points')
    args = parser.parse_args()
    return args

def infer(args):

    train_data = args.train_data
    test_data = args.test_data
    
    category = ["Aromatic", "HydrogenAcceptor", "HydrogenDonor", "Hydrophobic", "NegativeIon", "PositiveIon"]
    matching_category = {"Aromatic" : "Aromatic", 
        "Hydrophobic":"Hydrophobic",
        "HydrogenAcceptor":"HydrogenDonor",
        "HydrogenDonor":"HydrogenAcceptor",
        "NegativeIon": "PositiveIon",
        "PositiveIon": "NegativeIon" }
    
    matching_distance = {"Aromatic" : 7, 
        "Hydrophobic": 5,
        "HydrogenAcceptor": 4,
        "HydrogenDonor": 4,
        "NegativeIon": 5,
        "PositiveIon": 5 }
    feat_to_int = dict((c, i) for i, c in enumerate(category))
    int_to_feat = dict((i, c) for i, c in enumerate(category))

    if args.verbose:
        print('loading data')
    
    if len(train_data)>0:
        train_dataset = get_dataset(train_data,None,args,feat_to_int,int_to_feat,dump=False)  
    if len(test_data)>0:
        test_dataset = get_dataset(test_data,None,args,feat_to_int,int_to_feat,dump=False)    

    net= torch.load(args.model)
    net.eval()

    #output files for active learning
    dataset_file_train=None
    dataset_file_test=None
    train_file=None
    test_file=None
    if args.create_dataset:
        if len(train_data)>0:
            dataset_file_train=open(args.negative_output+"_train.txt",'w')
        if len(test_data)>0:
            dataset_file_test=open(args.negative_output+"_test.txt",'w')
    if args.output_pred:
        if len(train_data)>0:
            train_file=open(args.output+"_train.txt",'w')
        if len(test_data)>0:
            test_file=open(args.output+"_test.txt",'w')
    
    with torch.no_grad():
        if len(train_data)>0:
            if args.verbose:
                print("training set eval")
            predict(args,feat_to_int,int_to_feat,train_dataset,net,train_file,matching_category,matching_distance,dataset_file_train)
        if len(test_data)>0:
            if args.verbose:
                print("test set eval")
            predict(args,feat_to_int,int_to_feat,test_dataset,net,test_file,matching_category,matching_distance,dataset_file_test)


def check_pred(point_prediction,center,matching_category,matching_distance,protein_feats_df,dataset_file,ligand,protein,protein_coords):
    
    #take ony a subset of samples as the data generated is huge

    threshold_distance=matching_distance[point_prediction]
    category=matching_category[point_prediction]
    sub_df=protein_feats_df.loc[protein_feats_df['Feature'] == category]
    prot_feat_coords=sub_df.values[:,1:]
    center_expand=np.expand_dims(center,axis=0)
    if len(prot_feat_coords)==0:
        dataset_file.write("Not"+point_prediction+','+str(center[0])+','+str(center[1])+','+str(center[2])+','+ligand+','+protein+'\n')
        return
    distances=distance_matrix(center_expand,prot_feat_coords)
    #matching pharmacophore features should be witin a threshold distance
    if np.min(distances)>threshold_distance and np.random.uniform()<=0.1:
        dataset_file.write("Not"+point_prediction+','+str(center[0])+','+str(center[1])+','+str(center[2])+','+ligand+','+protein+'\n')
        return
    distances=distance_matrix(center_expand,protein_coords)
    # Hydrogen bond distance threshold at 1 angstrom
    if "Hydrogen" in point_prediction:
        if np.min(distances)<1 and np.random.uniform()<=0.1:
            dataset_file.write("Not"+point_prediction+','+str(center[0])+','+str(center[1])+','+str(center[2])+','+ligand+','+protein+'\n')
    # For the rest cannot be within 1.5 angstrom
    else:
        if "PositiveIon" in point_prediction:
            if np.min(distances)<1.5 and np.random.uniform()<=0.25:
                dataset_file.write("Not"+point_prediction+','+str(center[0])+','+str(center[1])+','+str(center[2])+','+ligand+','+protein+'\n')
        elif "NegativeIon" in point_prediction:
            if np.min(distances)<1.5 and np.random.uniform()<=0.10:
                dataset_file.write("Not"+point_prediction+','+str(center[0])+','+str(center[1])+','+str(center[2])+','+ligand+','+protein+'\n')
        #Very rarely aromatic predictions have steric problems for some reason
        elif "Aromatic" in point_prediction:
            if np.min(distances)<1.5:
                dataset_file.write("Not"+point_prediction+','+str(center[0])+','+str(center[1])+','+str(center[2])+','+ligand+','+protein+'\n')
        elif np.min(distances)<1.5 and np.random.uniform()<=0.1:
            dataset_file.write("Not"+point_prediction+','+str(center[0])+','+str(center[1])+','+str(center[2])+','+ligand+','+protein+'\n')
        else:
            return

def reduce_to_spheres(predictions,centers,matching_distance,matching_category,protein_feats_df,feat_to_int):
    spheres=np.copy(predictions)
    feat_to_sphere_ind={}
    feat_to_distances={}
    for category in matching_distance.keys():
        sub_df=protein_feats_df.loc[protein_feats_df['Feature'] == matching_category[category]]
        prot_feat_coords=sub_df.values[:,1:]
        if prot_feat_coords.shape[0]==0:
            spheres[index]=0
            continue
        distances=distance_matrix(centers,prot_feat_coords)
        threshold=matching_distance[category]
        #store distances for mask
        feat_to_distances[category]=distances
        #store indices to iterate over
        distances_min=np.min(distances,axis=0)
        indices=np.argwhere(distances_min<threshold)
        feat_to_sphere_ind[category]=indices
        distances_min=np.min(distances,axis=1)
        index=feat_to_int[category]
        spheres[:,index][distances_min > threshold] = 0.0
    return spheres, feat_to_sphere_ind, feat_to_distances

def get_coords_and_dimension(centers):

    x_coords=np.unique(centers[:,0])
    y_coords=np.unique(centers[:,1])
    z_coords=np.unique(centers[:,2])

    num_x=x_coords.shape[0]
    num_y=y_coords.shape[0]
    num_z=z_coords.shape[0]

    return [x_coords,y_coords,z_coords],[num_x,num_y,num_z]

def get_xyz(spheres,centers,feat_to_sphere_ind,feat_to_distances,feat_to_int,matching_category,matching_distance,prob_threshold):
    _,nums=get_coords_and_dimension(centers)

    num_x=nums[0]
    num_y=nums[1]
    num_z=nums[2]

    feat_to_coords={}
    #centers_grid=np.reshape(centers,(num_x,num_y,num_z,3))
    for category in feat_to_sphere_ind.keys():
        feat_to_coords[category]=[]
        distances=feat_to_distances[category]
        sphere_grid=spheres[:,feat_to_int[category]]
        sphere_grid=np.reshape(sphere_grid,(num_x,num_y,num_z))
        for ind in feat_to_sphere_ind[category]:
            distances_ind=distances[:,ind]
            distances_ind=np.reshape(distances_ind,(num_x,num_y,num_z))
            sphere_grid_ind=np.copy(sphere_grid)
            sphere_grid_ind[distances_ind>matching_distance[category]]=0
            binary_ind=(sphere_grid_ind>prob_threshold).astype(int)
            #extract connected components
            components, num_comp=label(binary_ind,return_num=True)
            #iterate over components, extract max position
            for i in range(1,num_comp+1):
                comp_grid=np.copy(sphere_grid_ind)
                comp_grid[components!=i]=0
                point_ind=np.argmax(comp_grid)
                #coord=centers_grid[point_ind]
                coord=centers[point_ind]
                feat_to_coords[category].append(coord)    
    return feat_to_coords

def cluster_xyz(feat_to_coords,distance_threshold=1.5):

    clustering = AgglomerativeClustering(n_clusters=None,compute_full_tree=True,linkage='average',distance_threshold=distance_threshold)
    for category in feat_to_coords.keys():
        if len(feat_to_coords[category])<2:
            continue
        final_xyz=[]
        coord_array=np.array(feat_to_coords[category])
        clustering.fit(coord_array)
        clusters=clustering.labels_
        for n in np.unique(clusters):
            cluster_center=coord_array[clusters==n].mean(axis=0)
            final_xyz.append(cluster_center)
        feat_to_coords[category]=final_xyz
    return feat_to_coords

def write_xyz(feat_to_coords,xyz_prefix,protein,top_dir):

    for category in feat_to_coords.keys():
        xyz_file_name='/'.join(protein.split('/')[:-1])+'/'+xyz_prefix+"_"+category+'.xyz'
        xyz_file_name=os.path.join(top_dir,xyz_file_name)
        xyz_file = open(xyz_file_name,'w')
        coord_set=set()
        for coords in feat_to_coords[category]:
            coord_set.add(tuple(coords))
        for coords in coord_set:
            xyz_file.write('H '+str(coords[0])+' '+str(coords[1])+' '+str(coords[2])+'\n')
        xyz_file.close()
    



def write_dx(centers,predicted,resolution,dx_file):
    def pretty_float(a):
        return "%0.5f" % a
    
    coords,nums=get_coords_and_dimension(centers)

    x_coords=coords[0]
    y_coords=coords[1]
    z_coords=coords[2]

    num_x=nums[0]
    num_y=nums[1]
    num_z=nums[2]
    
    origin_x=x_coords.min()
    origin_y=y_coords.min()
    origin_z=z_coords.min()

    origin=[pretty_float(origin_x),pretty_float(origin_y),pretty_float(origin_z)]
    dx_file.write("object 1 class gridpositions counts "+ str(num_x)+ " "+str(num_y)+" "+str(num_z)+"\n")
    dx_file.write("origin "+str(origin[0])+" "+str(origin[1])+" "+str(origin[2])+"\n")
    dx_file.write("delta "+str(resolution)+" 0 0\ndelta 0 "+str(resolution)+" 0\ndelta 0 0 "+str(resolution)+"\n")
    dx_file.write("object 2 class gridconnections counts "+ str(num_x)+ " "+str(num_y)+" "+str(num_z)+"\n")
    dx_file.write("object 3 class array type double rank 0 items [ "+ str(centers.shape[0])+"] data follows\n")
    count=0
    for prediction in predicted:
        dx_file.write(str(pretty_float(prediction)))
        count+=1
        if count%3==0:
            dx_file.write('\n')
        else:
            dx_file.write(" ")

def predict(args,feat_to_int,int_to_feat,dataset,net,output_file,matching_category,matching_distance,dataset_file=None):

    complexes=dataset.get_complexes()
    if args.verbose:
        print('num complexes:')
        print(len(complexes))
    tensor_shape = (args.batch_size,) + dataset.dims
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda', requires_grad=False)
    for complex in complexes:
        if args.verbose:
            print(complex)
        ligand=complex[0]
        protein=complex[1]
        if args.create_dataset or args.spherical:
            file=protein.split('nowat.pdb')[0]+'pharmfeats_obabel.csv'
            protein_feats_df=pd.read_csv(os.path.join(args.top_dir,file))
        #iterate and fill tensor till batch_size
        b=0
        centers=[]
        predicted=[]
        for center,grid in dataset.binding_site_grids(protein,ligand):
            input_tensor[b]=grid
            b+=1
            centers.append(center)
            if b==args.batch_size:
                outputs = net(input_tensor)
                sg_outputs = torch.sigmoid(outputs.detach().cpu())
                predicted += sg_outputs.tolist()
                #reset batch iteration
                b=0
        #if smaller than batch_size
        if b!=0:
            outputs = net(input_tensor[:b])
            sg_outputs = torch.sigmoid(outputs.detach().cpu())
            predicted += sg_outputs.tolist()
        intpredicted=np.round(predicted).astype(int)
        #if args.round_pred:
        #    predicted=np.round(predicted).astype(int)
        
        #output to file and create active learning set
        for center, predict,float_predict in zip(centers,intpredicted,predicted):
            point_predictions=[]
            for i in range(len(int_to_feat)):
                if predict[i]==1:
                    point_predictions.append(int_to_feat[i])
                # active learning dataset
                if float_predict[i]>args.dataset_threshold and args.create_dataset:
                    protein_coords=dataset.coordcache[protein].c.clone()
                    protein_coords=protein_coords.coords.tonumpy()
                    check_pred(point_predictions[-1],center,matching_category,matching_distance,protein_feats_df,dataset_file,ligand,protein,protein_coords)
            if args.output_pred:
                output_file.write(':'.join(point_predictions)+','+str(center[0])+','+str(center[1])+','+str(center[2])+','+ligand+','+protein+'\n')
        
        centers=np.array(centers)
        predictions=np.array(predicted)
        
        #spherical reduction
        if args.spherical:
            spheres, feat_to_sphere_ind, feat_to_distances=reduce_to_spheres(predictions,centers,matching_distance,matching_category,protein_feats_df,feat_to_int)
        
            #xyz output
            if args.xyz:
                feat_to_coords=get_xyz(spheres,centers,feat_to_sphere_ind,feat_to_distances,feat_to_int,matching_category,matching_distance,args.prob_threshold)
                feat_to_coords=cluster_xyz(feat_to_coords,args.clus_threshold)
                write_xyz(feat_to_coords,args.prefix_xyz,protein,args.top_dir)
        #output dx files of predictions
        if args.create_dx:     
            if args.round_pred:
                predictions=np.array(intpredicted)
            if args.spherical:
                predictions=spheres
            for i in range(len(int_to_feat)):
                dx_file_name='/'.join(protein.split('/')[:-1])+'/'+args.prefix_dx+"_"+int_to_feat[i]+'.dx'
                dx_file_name=os.path.join(args.top_dir,dx_file_name)
                dx_file=open(dx_file_name,'w')
                write_dx(centers,predictions[:,i],dataset.resolution,dx_file)
                dx_file.close()

if __name__ == '__main__':

    args = parse_arguments()
    infer(args)