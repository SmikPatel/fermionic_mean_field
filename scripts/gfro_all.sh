#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --job-name=GFRO_ALL
#SBATCH --output=GFRO_ALL_OUTPUT.txt

cd ..

module load intelpython3

source activate meanfield_env

for mol in h2 h4 lih beh2 h2o nh3; do
python main_gfro_resumable.py $mol 100 None
done