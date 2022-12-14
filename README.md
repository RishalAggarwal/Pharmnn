# Pharmacophore prediction

Train_pharmnn.py - Model training and pickling data

Dataset.py - Pytorch Dataset classes for original and retraining with sampled negatives

clean_pdb.py - Remove waters and hetero atoms from pdb file

model.py - CNN model class

pharm_rec.py - featurize pdb files to apply heuristics for active learning

pharmaco_env.yaml - conda environment

setup_splits.py - setup splits for grid generation and protein and ligand dataset

submit_2.slurm - model training on gpu cluster

data/ - dataset folder - files are too large to be shared on github

models/ - model folder - files are too large to be shared on github

example usage:

``` python train_pharmnn.py --train_data data/chemsplit_train0_with_ligand.pkl --test_data data/chemsplit_test0_with_ligand.pkl  --wandb_name iter1_chemsplit0 --negative_data data/iter1_chemsplit_0_negatives_train.txt --batch_size 128 ```
