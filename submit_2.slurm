#!/bin/bash

#job name
#SBATCH --job pharmnn_training
#SBATCH --partition dept_gpu
#SBATCH --gres=gpu:1

#SBATCH --exclude g001,g011,g019

#SBATCH --mail-user=ria43@pitt.edu
#SBATCH --mail-type=ALL

# directory name where job will be run (on compute node)
#job_dir="${user}_${SLURM_JOB_ID}.dcb.private.net"

# creating directory on /scr folder of compute node
#mkdir /scr/$job_dir

# put date and time of starting job in a file
#date > date.txt

# put hostname of compute node in a file
#hostname > hostname.txt

# copy files on exit or interrupt
# make sure this is before your main program for it to always run on exit
#trap "echo 'copying files'; rsync -avz * ${SLURM_SUBMIT_DIR}" EXIT

# copy the submit file (and all other related files/directories)
#rsync -a ${SLURM_SUBMIT_DIR}/*.pkl /scr/${job_dir}


source activate phramnn
module load cuda/11.5
#python ./train_pharmnn.py --train_data data/chemsplit_train0.pkl --test_data data/chemsplit_test0.pkl  --wandb_name default_chemsplit0_large_256 --grid_dimension 15.5  --expand_width 0 --model models/default_chemsplit0_large_256_last_model.pkl --lr 0.00001
python ./train_pharmnn.py --train_data data/chemsplit_train0_with_ligand.pkl --test_data data/chemsplit_test0_with_ligand.pkl  --wandb_name iter1_chemsplit0 --negative_data data/iter1_chemsplit_0_negatives_train.txt --batch_size 128
