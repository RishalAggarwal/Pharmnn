#!/bin/bash

#job name
#SBATCH --job pharmnn_dataset
##SBATCH --partition dept_gpu
##SBATCH --gres=gpu:1
##SBATCH --exclude g001,g012,g013,g019,g121

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

export PYTHONUNBUFFERED=TRUE
source activate phramnn
#module load cuda/11.5
python ./train_pharmnn.py --train_data data/chemsplit_train1_with_ligand.txt --test_data data/chemsplit_test1_with_ligand.txt --pickle_only --top_dir data/
#python inference.py --train_data data/chemsplit_train0_with_ligand.pkl  --create_dataset --verbose --model models/default_chemsplit0_best_model.pkl --negative_output data/iter1_chemsplit_0_negatives
