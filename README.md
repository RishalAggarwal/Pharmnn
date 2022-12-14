# Pharmacophore prediction

Train_pharmnn.py - Model training and pickling data

Dataset.py - Pytorch Dataset classes for original and retraining with sampled negatives

clean_pdb.py - Remove waters and hetero atoms from pdb file

model.py - CNN model class

pharm_rec.py - featurize pdb files to apply heuristics for active learning

pharmaco_env.yaml - conda environment

setup_splits.py - setup splits for grid generation and protein and ligand dataset

submit_2.slurm - model training on gpu cluster

data/ - dataset folder

models/ - model folder

