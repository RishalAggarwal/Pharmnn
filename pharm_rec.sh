#!/bin/bash

#job name
#SBATCH --job active_dataset
#SBATCH --partition dept_cpu
#SBATCH --exclude g001,g012,g013,g019,g121

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
#python ./train_pharmnn.py --train_data /scr/${job_dir}/bigchemsplit_train0.pkl --test_data /scr/${job_dir}/bigchemsplit_test0.pkl  
python pharm_rec.py ./data/refined-set/
python pharm_rec.py ./data/v2020-other-PL/