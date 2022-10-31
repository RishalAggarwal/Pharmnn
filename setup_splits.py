import os

for file in os.listdir('data_csvs'):
    if 'data' not in file:
        f1=open(os.path.join('data_csvs',file),'r')
        f2=open(os.path.join('data_csvs',file.split('.txt')[0]+'_with_ligand.txt'),'w')
        for line in f1:
            ligand=line.split(',')[-1].replace('_protein_nowat.pdb\n','_ligand.sdf')
            new_line=','.join(line.split(',')[:4])+','+ligand+','+line.split(',')[-1]
            f2.write(new_line)


