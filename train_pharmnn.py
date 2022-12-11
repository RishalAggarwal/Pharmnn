#!/usr/bin/env python3

import torch
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from sklearn.metrics import confusion_matrix
import itertools


import molgrid
try:
    from molgrid.openbabel import pybel
except ImportError:
    from openbabel import pybel

pybel.ob.obErrorLog.StopLogging() #without this wandb will deadlock when ob fills up the write buffer

def parse_arguments():
    parser = argparse.ArgumentParser('Train a CNN on GIST data to predict pharmacophore feature.')
    parser.add_argument('--wandb_name',required=False,help='data to train with',default=None)
    parser.add_argument('--train_data',required=True,help='data to train with',default="data_train_pdb.txt")
    parser.add_argument('--test_data',default="",help='data to test with')
    parser.add_argument('--pickle_only',help="Create pickle files of the data only; don't train",action='store_true')
    parser.add_argument('--top_dir',default='.',help='root directory of data')
    parser.add_argument('--batch_size',default=256,type=int,help='batch size')
    parser.add_argument('--epochs',default=265,type=int,help='number epochs')
    parser.add_argument('--steplr',default=50,type=int,help='when to step the learning rate')
    parser.add_argument('--patience',default=35,type=int,help='when to step the learning rate')
    parser.add_argument('--eval_only',help="only evaluate the model, no training",action='store_true')
    parser.add_argument('--model',type=str,help='model to test with',default=None)
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
    parser.add_argument('--prob_threshold', type=float,default=0.5,help='probability threshold for metrics')
    parser.add_argument('--use_gist', type=int,default=0,help='use gist grids')
    parser.add_argument('--rotate', type=int,default=1,help='random rotations of pdb grid')
    parser.add_argument('--use_se3', type=int,default=0,help='use se3 convolutions')
    parser.add_argument('--seed',default=42,type=int,help='random seed')
    parser.add_argument('--verbose',help="verbse complex working on",action='store_true')

    args = parser.parse_args()
    return args

def log_metrics(prefix, labels, predicts,epoch,category,feat_to_int,int_to_feat,verbose):
    '''Given true labels and unrounded predictions calculate and log metrics.
    These should be lists of 6-vectors'''
    labels = np.array(labels)
    predicts = np.array(predicts)
    
    metrics = {'epoch':epoch}
    f1_total_test=0
    f1_total_train=0
    for cname in category:
        i = feat_to_int[cname]
        L = labels[:,i]
        P = predicts[:,i]
        # original method for creating imbalanced dataset is not viable (each protein has specific labels)
        # will make pass for now
        try:
            metrics[prefix+' '+cname+' Accuracy'] = sklearn.metrics.accuracy_score(L, (np.array(P)>args.prob_threshold).astype(int))
            metrics[prefix+' '+cname+' Precision'] = sklearn.metrics.precision_score(L, (np.array(P)>args.prob_threshold).astype(int))
            metrics[prefix+' '+cname+' Recall'] = sklearn.metrics.recall_score(L, (np.array(P)>args.prob_threshold).astype(int))
            metrics[prefix+' '+cname+' F1'] = sklearn.metrics.f1_score(L, (np.array(P)>args.prob_threshold).astype(int))
            if 'Test' in prefix:
                f1_total_test+=metrics[prefix+' '+cname+' F1']
            else:
                f1_total_train+=metrics[prefix+' '+cname+' F1']
            metrics[prefix+' '+cname+' AUC'] = sklearn.metrics.roc_auc_score(L, P)
        except ValueError:
            pass
    if f1_total_test != 0:
        metrics['Total Test F1'] = f1_total_test
    if f1_total_train != 0:
        metrics['Total Train F1'] = f1_total_train
    if args.eval_only:
        #confusion matrix
        predicts_conf=(np.array(predicts)>args.prob_threshold).astype(int)
        cm=np.zeros((len(category),len(category)))
        cm_neg=np.zeros((len(category),len(category)))
        for cname1 in category:
            i=feat_to_int[cname1]
            for cname2 in category:
                j=feat_to_int[cname2]
                label_matrix=labels[:,i]
                if i!=j:
                    #account for when the label for j is present
                    label_matrix=labels[:,i]-labels[:,j]
                    #sometimes j is present but i isnt
                    label_matrix[label_matrix<0]=0
                #Me being lazy, there is probably a better way to get true positives
                matrix=confusion_matrix(label_matrix,predicts_conf[:,j])
                cm[i][j]+=matrix[0][0]
                cm_neg[i][j]+=matrix[1][1]
        data=[]
        data_neg=[]
        for j, i in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                pred_dict = int_to_feat[i]
                true_dict = int_to_feat[j]
                data.append([pred_dict, true_dict, cm[j,i]])
                data_neg.append([pred_dict, true_dict, cm_neg[j,i]])
        wandb.log({prefix+"confusion_matrix_positive" : wandb.Table(
                columns=['Predicted', 'Actual', 'Count'],
                data=data)})
        wandb.log({prefix+"confusion_matrix_negative" : wandb.Table(
                columns=['Predicted', 'Actual', 'Count'],
                data=data_neg)})

    if verbose:
        print(metrics)
    
    wandb.log(metrics)
    

    return metrics
    
def get_dataset(fname, args,feat_to_int,int_to_feat,dump=True):
    '''Create a dataset.  If a pkl file is not passed, create one for faster loading later'''
    if fname.endswith('.pkl'):
        dataset = pickle.load(open(fname,'rb'))
        cache=None
        coordcache=None
        classcnts=None
        #cache is a list of datapoints
        if isinstance(dataset,list):
            cache=dataset[0]
            coordcache=dataset[1]
            classcnts=dataset[2]
        #if loaded object is pharmacophoredataset and not cache
        else:
            cache=dataset.cache
            classcnts=dataset.classcnts
            coordcache=dataset.coordcache
        dataset = PharmacophoreDataset(txt_file=fname,feat_to_int=feat_to_int,int_to_feat=int_to_feat,top_dir=args.top_dir, grid_dimension=args.grid_dimension, rotate=args.rotate, use_gist=args.use_gist,cache=cache,classcnts=classcnts,coordcache=coordcache)
        #dataset.rotate = args.rotate
        #TODO: factor out parameters that shouldn't be pickled, basically just pickle the cache (create a new class)
        #dataset.use_gist = args.use_gist
        #dataset.gmaker = MyGridMaker(resolution=0.5, dimension=args.grid_dimension) 
        #dataset.dims = dataset.gmaker.g.grid_dimensions(molgrid.defaultGninaReceptorTyper.num_types())        
        return dataset
    else:
        dataset = PharmacophoreDataset(txt_file=fname,feat_to_int=feat_to_int,int_to_feat=int_to_feat,top_dir=args.top_dir, grid_dimension=args.grid_dimension, rotate=args.rotate, use_gist=args.use_gist)
        if dump:
            prefix,ext = os.path.splitext(fname)
            pickle.dump([dataset.cache,dataset.coordcache,dataset.classcnts], open(prefix+'.pkl','wb'))
        return dataset

def train(args):
    
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

    wandb.init(project="pharmnn", config=args, name=args.wandb_name)

    train_data = args.train_data
    test_data = args.test_data
    if not test_data: # infer test file name from train file name - makes wandb sweep easier
        test_data = train_data.replace('train','test')

    #one-hot encoder vectors
    category = ["Aromatic", "HydrogenAcceptor", "HydrogenDonor", "Hydrophobic", "NegativeIon", "PositiveIon"]
    feat_to_int = dict((c, i) for i, c in enumerate(category))
    int_to_feat = dict((i, c) for i, c in enumerate(category))

        
    #Creation of test set/loader (individual system)


    dataset1 = get_dataset(train_data,args,feat_to_int,int_to_feat)
    trainloader = DataLoader(dataset1, batch_size=args.batch_size, num_workers=0, shuffle=True,drop_last=False)

    dataset2 = get_dataset(test_data,args,feat_to_int,int_to_feat)
    testloader = DataLoader(dataset2, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)


    if args.pickle_only:
        sys.exit(0)

    #Training
    if args.model:
        net= torch.load(args.model)
    else:
        net = GISTNet(args)
        net.apply(weights_init)
    if args.verbose:
        print(net)
    net.to('cuda')
    wandb.watch(net)

    best_f1=0

    paramcnt = sum([np.prod(p.size()) for p in  filter(lambda p: p.requires_grad, net.parameters())])
    wandb.log({'Parameters': paramcnt})
    if args.verbose:
        print("Parameters",paramcnt)

    #calculate weights of classes
    pos_weight = [(len(dataset1)-ccnt)/ccnt for ccnt in dataset1.classcnts] 
    criterion = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.tensor(pos_weight).to('cuda'))

    if args.solver == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    clip_value = args.clip
    #TODO: update learning rate at plateau
    change_lr = StepLR(optimizer, step_size = 1, gamma=0.1)
    steplr = args.steplr
    scheduler = ReduceLROnPlateau(optimizer, 'max',patience=args.patience,min_lr=1e-6)
    
    if args.verbose:
        print("starting training")
    #TODO: early stopping
    num_epochs=args.epochs
    if args.eval_only:
        num_epochs=1
    for epoch in range(num_epochs):
        if args.eval_only:
            net.eval()
        net.train()
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
            if not args.eval_only:
                loss.backward()
                
                wandb.log({'Training Loss': loss})
                if i % 100 == 0 and args.verbose:
                    print('Training Loss', epoch, i, loss.item())
                if clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
                optimizer.step()

            
            sg_outputs = torch.sigmoid(outputs.detach().cpu()) # push through a sigmoid layer just for accuracy calculations
            predicted += sg_outputs.tolist()
            labels += data['label'].cpu().tolist()

        wandb.log({'Epoch':epoch})
        train_metrics = log_metrics('Train',labels,predicted,epoch,category,feat_to_int,int_to_feat,args.verbose)
        if args.verbose:
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
        if args.verbose:
            print('Learning rate is ', scheduler.get_lr())
        
        test_metrics=log_metrics('Test',test_labels,test_predict,epoch,category,feat_to_int,int_to_feat,args.verbose)
        wandb.log({'Learning Rate': optimizer.param_groups[0]['lr']})

        #TODO: update lr on plateau
        scheduler.step(test_metrics['Total Test F1'])
        '''if epoch != 0 and (epoch % steplr) == 0:
            change_lr.step()
            steplr = steplr//2  #each step has twice as few iterations'''
        if test_metrics['Total Test F1']>best_f1 and not args.eval_only:
            best_f1=test_metrics['Total Test F1']
            wandb.run.summary["Best Test F1"] = best_f1
            torch.save(net, "models/"+ wandb.run.name + "_best_model.pkl")
        if not args.eval_only:
            torch.save(net, "models/"+ wandb.run.name + "_last_model.pkl")  
    if args.verbose:
        print("finished training")

if __name__ == '__main__':

    args = parse_arguments()
    train(args)
