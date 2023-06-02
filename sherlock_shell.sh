#!/bin/bash 
#SBATCH --job-name=test
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/grawoods/.out/%j.out
#SBATCH --error=/scratch/users/grawoods/.out/%j.out

ml python/3.9.0
ml py-numpy/1.20.3_py39
ml py-scikit-learn/1.0.2_py39
ml py-scipy/1.10.1_py39
ml viz
ml py-matplotlib/3.4.2_py39

python3 sherlock_executable.py --datadir="$SCRATCH"/SHERLOCK_TEST/200516_M200212/ --maxfiles=2 --outdir="$SCRATCH"/Results/