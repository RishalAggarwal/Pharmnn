'''
Takes a PDB file and removes hetero atoms from its structure.
First argument is path to original file, second argument is path to generated file
'''
from Bio.PDB import PDBParser, PDBIO, Select
import Bio
import os
import sys

class NonHetSelect(Select):
    def accept_residue(self, residue):
        return 1 if Bio.PDB.Polypeptide.is_aa(residue,standard=True) else 0

def clean_pdb(input_file,output_file):
    pdb = PDBParser().get_structure("protein", input_file)
    io = PDBIO()
    io.set_structure(pdb)
    io.save(output_file, NonHetSelect())
    
if __name__ == '__main__':
    count=0
    for d in os.listdir(sys.argv[1]):
        if len(d)!=4:
            continue
        count+=1
        print(count,d)
        clean_pdb(os.path.join(sys.argv[1],d,d+"_protein.pdb"),os.path.join(sys.argv[1],d,d+"_protein_nowat.pdb"))