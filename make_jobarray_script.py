from pathlib import Path
import argparse

def generate_jobscript(animal_dir, num_sessions):
    fpath = f"_jobarray_oversessions.sh"
    with open(fpath, 'w') as f:
        sbatch_commands = [
            '#!/bin/bash \n',
            f'#SBATCH --job-name={str(animal_dir.stem)}\n',
            '#SBATCH --time=3:00:00\n',
            '#SBATCH --mem=20G\n',
            f'#SBATCH --output=/scratch/users/grawoods/.out/{str(animal_dir.stem)}_%j.out\n',
            f'#SBATCH --error=/scratch/users/grawoods/.out/{str(animal_dir.stem)}_%j.out\n',
            '#SBATCH --mail-type ARRAY_TASKS\n',
            f'#SBATCH --array=0-{num_sessions-1}\n'
        ]
        f.writelines(sbatch_commands)
        f.write('\n')

        f.write(f'path_to_datadir={animal_dir}\n')
        f.write('list_of_dirs=($(find $path_to_datadir -mindepth 1 -type d))\n')
        f.write('num_dirs=${#list_of_dirs[@]}\n')
        f.write('echo Found $num_dirs subdirectories!\n')
        f.write('\n')

        f.write('session_dir=${list_of_dirs[$SLURM_ARRAY_TASK_ID]}\n')
        f.write('echo Array task ID: $SLURM_ARRAY_TASK_ID, for session at:\n')
        f.write('echo $session_dir\n')
        f.write('echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID\n')
        f.write('echo SLURM_JOB_ID: $SLURM_JOB_ID\n')
        f.write('echo SLURM_TASK_ID: $SLURM_TASK_ID\n')
        f.write('\n')
        
        ml_commands = [
            'ml python/3.9.0\n',
            'ml py-numpy/1.20.3_py39\n',
            'ml py-scikit-learn/1.0.2_py39\n',
            'ml py-scipy/1.10.1_py39\n',
            'ml viz\n',
            'ml py-matplotlib/3.4.2_py39\n'
        ]
        f.writelines(ml_commands)
        f.write('\n')

        f.write(f'python3 sherlock_executable \\\n')
        f.write(f'    --datadir={animal_dir} \\\n')
        f.write('    --maxfiles=-1 \\\n')
        f.write(f'    --outdir={str(animal_dir.parent)}/Results\n')
        f.write('\n')
    return fpath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate array job script for given Animal ID')
    parser.add_argument('--animal_dir', type=str, help='Path to animal directory')

    args = parser.parse_args()
    animal_dir = Path(args.animal_dir)
    recording_dir = [x for x in animal_dir.iterdir() if x.is_dir()]
    num_sessions = len(recording_dir)

    fpath = generate_jobscript(animal_dir=animal_dir, num_sessions=num_sessions)
    print(fpath)
