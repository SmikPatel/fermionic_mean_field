#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --job-name=GFRO_ALL_RESUME1
#SBATCH --output=GFRO_ALL_RESUME1_OUTPUT.txt

cd ..

module load intelpython3

source activate meanfield_env

python main_gfro_resumable.py nh3 100 34