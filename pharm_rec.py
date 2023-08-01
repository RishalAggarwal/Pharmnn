import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmarts,ForwardSDMolSupplier,MolFromPDBFile, AddHs,rdmolfiles
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField, OptimizeMolecule
from glob import glob
try:
    from molgrid.openbabel import pybel
except ImportError:
    from openbabel import pybel
import sys
import os

def get_mol_pharm(rdmol,obmol):
    # define smarts strings
    #use both openbabel and rdkit to get the smarts matches
    smarts={}
    smarts['Aromatic']=["a1aaaaa1", "a1aaaa1"]
    smarts['PositiveIon'] = ['[+,+2,+3,+4]',"[$(C(N)(N)=N)]", "[$(n1cc[nH]c1)]"]
    smarts['NegativeIon'] = ['[-,-2,-3,-4]',"C(=O)[O-,OH,OX1]"]
    smarts['HydrogenAcceptor']=["[#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4])&!$(N=C([C,N])N)]","[$([O])&!$([OX2](C)C=O)&!$(*(~a)~a)]"]
    smarts['HydrogenDonor']=["[#7!H0&!$(N-[SX4](=O)(=O)[CX4](F)(F)F)]", "[#8!H0&!$([OH][C,S,P]=O)]","[#16!H0]"]
    smarts['Hydrophobic']=["a1aaaaa1","a1aaaa1","[$([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(**[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]",
                            "[$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
                            "[CH2X4,CH1X3,CH0X2]~[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]","[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
                            "[$([S]~[#6])&!$(S~[!#6])]"]

    atoms = obmol.atoms
    pharmit_feats={}
    for key in smarts.keys():
        for smart in smarts[key]:
            obsmarts = pybel.Smarts(smart) # Matches an ethyl group
            matches = obsmarts.findall(obmol)
            smarts_mol=MolFromSmarts(smart)
            rd_matches=rdmol.GetSubstructMatches(smarts_mol,uniquify=True)
            for match in matches:
                positions=[]
                for idx in match:
                    positions.append(np.array(atoms[idx-1].coords))
                positions=np.array(positions).mean(axis=0)
                if key in pharmit_feats.keys():
                    pharmit_feats[key].append(positions)
                else:
                    pharmit_feats[key]=[positions]
            for match in rd_matches:
                positions=[]
                for idx in match:
                    positions.append(np.array(atoms[idx].coords))
                positions=np.array(positions).mean(axis=0)
                if key in pharmit_feats.keys():
                    pharmit_feats[key].append(positions)
                else:
                    pharmit_feats[key]=[positions]

    return pharmit_feats

def pharm_rec(file):
    rdmol=rdmolfiles.MolFromPDBFile(file,sanitize=True) 
    obmol =next(pybel.readfile("pdb", file))
    pharmit_feat=get_mol_pharm(rdmol,obmol)
    f=open(file.split('_nowat.pdb')[0]+'_pharmfeats_obabel.csv','w')
    f.write('Feature,x,y,z\n')
    for key in pharmit_feat.keys():
        for position in pharmit_feat[key]:
            f.write(key+','+str(position[0])+','+str(position[1])+','+str(position[2])+'\n')
    f.close()


if __name__ == '__main__':
    data_dir=sys.argv[1]
    os.chdir(data_dir)
    pdbs=glob('./*_nowat.pdb')
    for pdb in pdbs:
        print(pdb)
        pharm_rec(pdb)
