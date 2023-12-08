#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --job-name=MEANFIELD_NH3_RESUME2
#SBATCH --output=MEANFIELD_NH3_RESUME2_OUTPUT.txt

cd ..

module load intelpython3

source activate meanfield_env

python main_meanfield_resumable.py nh3 100 35

