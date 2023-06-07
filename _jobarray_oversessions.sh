#!/bin/bash 
#SBATCH --job-name=WT_1
#SBATCH --time=3:00:00
#SBATCH --mem=20G
#SBATCH --output=/scratch/users/grawoods/.out/WT_1_%j.out
#SBATCH --error=/scratch/users/grawoods/.out/WT_1_%j.out
#SBATCH --mail-type ARRAY_TASKS
#SBATCH --array=0-13

path_to_datadir=/Users/GraceWoods/Desktop/Neural_Analysis/neural_analysis_python/WT_1
list_of_dirs=($(find $path_to_datadir -mindepth 1 -type d))
num_dirs=${#list_of_dirs[@]}
echo Found $num_dirs subdirectories!

session_dir=${list_of_dirs[$SLURM_ARRAY_TASK_ID]}
echo Array task ID: $SLURM_ARRAY_TASK_ID, for session at:
echo $session_dir
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID
echo SLURM_JOB_ID: $SLURM_JOB_ID
echo SLURM_TASK_ID: $SLURM_TASK_ID

ml python/3.9.0
ml py-numpy/1.20.3_py39
ml py-scikit-learn/1.0.2_py39
ml py-scipy/1.10.1_py39
ml viz
ml py-matplotlib/3.4.2_py39

python3 sherlock_executable \
    --datadir=/Users/GraceWoods/Desktop/Neural_Analysis/neural_analysis_python/WT_1 \
    --maxfiles=-1 \
    --outdir=/Users/GraceWoods/Desktop/Neural_Analysis/neural_analysis_python/Results

