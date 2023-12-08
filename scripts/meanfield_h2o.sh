#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --job-name=MEANFIELD_H2O
#SBATCH --output=MEANFIELD_H2O_OUTPUT.txt

cd ..

module load intelpython3

source activate meanfield_env

python main_meanfield_resumable.py h2o 100 None

