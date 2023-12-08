"""
purpose : do meanfield decomposition of Htbt until np.sum(Htbt * Htbt) < 1e-6 with equal splitting of spin-orbitals

CLA     : python main_meanfield.py (moltag) (bonustag) 
"""

import numpy as np
import time

from utils_meanfield import (
    get_fragment,
    obtain_meanfield_fragment
)

from utils_saveload import (
    load_hamiltonian_tensors,
    save_params,
    output_log_filename
)

# load in command line arguments (molecule and experiment id) and some setup 

import sys

moltag    = sys.argv[1]
bonustag  = sys.argv[2]
methodtag = 'meanfield'
outputtag = output_log_filename(moltag, methodtag, bonustag)

# load in Hamiltonian chemist tensors

Hobt, Htbt = load_hamiltonian_tensors(moltag)
N          = Hobt.shape[0] 

S1 = list(range(N//2))
S2 = list(range(N//2, N))

N1 = len(S1)
N2 = len(S2)

# set up params and norms lists and print initial norm

params = list()
norms  = list()

current_norm = np.sum(Htbt * Htbt)
norms.append(current_norm)

with open(outputtag, 'a') as f:
    print(f'\nInitial Norm : {current_norm}\n', file=f)

# do meanfield decomposition with convergence criterion tol=1e-6

numtag        = 10000
tol           = 1e-6
num_fragments = 0

for k in range(numtag):
    # end decomposition if remaining norm is less than 1e-6
    if current_norm < tol:
        break

    # obtain fragment parameters and remove fragment from Htbt
    sol   = obtain_meanfield_fragment(Htbt, S1, S2)
    Htbt -= get_fragment(sol.x, S1, S2, N1, N2)

    # save current parameters and current norm
    params.append(sol.x)

    current_norm = np.sum(Htbt * Htbt)
    norms.append(current_norm)

    # print output
    num_fragments += 1
    with open(outputtag, 'a') as f:
        print(f'fragment count : {num_fragments}, current norm : {np.round(current_norm, 7)}\n', file=f)

with open(outputtag, 'a') as f:
    print(f"""
        Molecule            : {moltag}
        Method              : {methodtag}
        Number of Fragments : {num_fragments}
        Final Norm          : {np.sum(Htbt * Htbt)} 
    """, file=f)

save_params(params, norms, moltag, methodtag, bonustag)