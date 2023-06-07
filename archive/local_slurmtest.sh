#!/bin/bash 
#SBATCH --job-name=test
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/grawoods/.out/%j.out
#SBATCH --error=/scratch/users/grawoods/.out/%j.out
#SBATCH --array=1-XXX

#getting datadir
path_to_datadir=/Users/GraceWoods/Desktop/Neural_Analysis/neural_analysis_python/WT_1
# # python3 getdatadirs.py --datadir=path_to_datadir
# cd path_to_datadir
# list_of_dirs=($(find . -type d -mindepth 1)) # in curr directory; will print (./dir_of_recording/)
# # note, right now 'find' produces the current working directory so is overcounting... how to adjust? <- fixed with mindepth
# list_of_dirs=($(find ~+ -type d -mindepth 1)) # will print entire dir of dir_of_recording

#OR
list_of_dirs=($(find $path_to_datadir -type d -mindepth 1)) #-mindepth 1)) # which will print out the entire dir of dir_of_recording
# echo ${list_of_dirs[*]}

num_dirs=${#list_of_dirs[@]} # this will define the range of --array in the SBATCH preamble
# question: can we do all of this before the #SBATCH --array=1-XXX command? 
echo Found $num_dirs subdirectories!

#$SLURM_ARRAY_TASK_ID
file=${list_of_dirs[0]} #$SLURM_ARRAY_TASK_ID-1]}
echo $file

file2=${list_of_dirs[1]} #$SLURM_ARRAY_TASK_ID-1]}
echo $file2

## it looks like there's a diff between
## 'bash local_slurmtest.sh': does not access all of the subdirs
## and 'zsh local_slurmtest.sh': accesses all of the subdirs

# question: keeping track of which directories have been analyzed in case job dies?