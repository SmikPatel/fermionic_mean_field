#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --job-name=MEANFIELD_UPTO_BEH2
#SBATCH --output=MEANFIELD_UPTO_BEH2_OUTPUT.txt

cd ..

module load intelpython3

source activate meanfield_env

for mol in h2 h4 lih beh2; do
python main_meanfield_resumable.py $mol 100 None
done