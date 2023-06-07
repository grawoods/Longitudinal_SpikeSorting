#!/bin/bash 
#SBATCH --job-name=test
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/grawoods/.out/%j.out
#SBATCH --error=/scratch/users/grawoods/.out/%j.out
#SBATCH --array=1-XXX

ml python/3.9.0
ml py-numpy/1.20.3_py39
ml py-scikit-learn/1.0.2_py39
ml py-scipy/1.10.1_py39
ml viz
ml py-matplotlib/3.4.2_py39

#getting datadir
path_to_datadir=$SCRATCH

animal_ids=(AD_M1  AD_M2  AD_M3  AD_M4  WT_M1  WT_M3  WT_M4  WT_PANDEMIC)

# python3 getdatadirs.py --datadir=path_to_datadir
cd path_to_datadir
list_of_dirs=($(find . -type d -mindepth 1)) # in curr directory; will print (./dir_of_recording/)
# note, right now 'find' produces the current working directory so is overcounting... how to adjust? <- fixed with mindepth
list_of_dirs=($(find ~+ -type d -mindepth 1)) # will print entire dir of dir_of_recording

#OR
list_of_dirs=($(find path_to_datadir -type d -mindepth 1)) # which will print out the entire dir of dir_of_recording

num_dirs=${#list_of_dirs[@]} # this will define the range of --array in the SBATCH preamble
# question: can we do all of this before the #SBATCH --array=1-XXX command? 


#$SLURM_ARRAY_TASK_ID
file=${list_of_dirs[$SLURM_ARRAY_TASK_ID-1]}
echo $file

python3 sherlock_executable.py --datadir="$SCRATCH"/SHERLOCK_TEST/200516_M200212/ --maxfiles=2 --outdir="$SCRATCH"/Results/


# python3 sherlock_executable.py --datadir="$SCRATCH"/SHERLOCK_TEST/200516_M200212/ --maxfiles=2 --outdir="$SCRATCH"/Results/