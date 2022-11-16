import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmarts,ForwardSDMolSupplier,MolFromPDBFile, AddHs,rdmolfiles
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField, OptimizeMolecule
from glob import glob
import sys
import os

def get_mol_pharm(mol):
    # define smarts strings
    smarts={}
    smarts['PositiveIon'] = ['[+,+2,+3,+4]','[$(C*)](=,-N)N','C(N)(N)=N','[nH]1cncc1']
    smarts['NegativeIon'] = ['[-,-2,-3,-4]','[S,P,C](=O)[O-,OH,OX1]','c1[nH1]nnn1','c1nn[nH1]n1','C(=O)N[OH1,O-,OX1]','CO(=N[OH1,O-])',
                            '[$(N-[SX4](=O)(=O)[CX4](F)(F)F)]']
    smarts['Aromatic']=["a1aaaaa1", "a1aaaa1"]
    smarts['HydrogenAcceptor']=["[#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4])&!$(N=C([C,N])N)]","[$([O])&!$([OX2](C)C=O)&!$(*(~a)~a)]"]
    smarts['HydrogenDonor']=["[#7!H0&!$(N-[SX4](=O)(=O)[CX4](F)(F)F)]", "[#8!H0&!$([OH][C,S,P]=O)]","[#16!H0]"]
    smarts['Hydrophobic']=["a1aaaaa1","a1aaaa1","[$([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(**[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]",
                            "[$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
                            "[CH2X4,CH1X3,CH0X2]~[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]","[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
                            "[$([S]~[#6])&!$(S~[!#6])]"]

    conf = mol.GetConformer()
    pharmit_feats={}
    for key in smarts.keys():
        for smart in smarts[key]:
            smarts_mol=MolFromSmarts(smart)
            matches=mol.GetSubstructMatches(smarts_mol,uniquify=True)
            for match in matches:
                positions=[]
                for idx in match:
                    positions.append(np.array(list(conf.GetAtomPosition(idx))))
                positions=np.array(positions).mean(axis=0)
                if key in pharmit_feats.keys():
                    pharmit_feats[key].append(positions)
                else:
                    pharmit_feats[key]=[positions]

    return pharmit_feats

def pharm_rec(file):
    mol=rdmolfiles.MolFromPDBFile(file,sanitize=False) 
    pharmit_feat=get_mol_pharm(mol)
    f=open(file.split('_protein_nowat.pdb')[0]+'_protein_pharmfeats.csv','w')
    f.write('Feature,x,y,z\n')
    for key in pharmit_feat.keys():
        for position in pharmit_feat[key]:
            f.write(key+','+str(position[0])+','+str(position[1])+','+str(position[2])+'\n')
    f.close()


if __name__ == '__main__':
    data_dir=sys.argv[1]
    os.chdir(data_dir)
    pdbs=glob('./*/*_protein_nowat.pdb')
    for pdb in pdbs:
        print(pdb)
        pharm_rec(pdb)
