"""
additional useful subroutines for processing of results
"""

import numpy as np

from openfermion import (
    FermionOperator,
    get_sparse_operator,
    get_ground_state,
    variance
)

def chem_ten2op(obt, tbt, N):
    """
    convert one-body-tensor and or two-body-tensor into FermionOperator form
    """
    op1 = FermionOperator()
    op2 = FermionOperator()
    for p in range(N):
        for q in range(N):
            term = ((p,1), (q,0))
            coef = obt[p,q]
            op1 += FermionOperator(term, coef)
            for r in range(N):
                for s in range(N):
                    term = ((p,1), (q,0), (r,1), (s,0))
                    coef = tbt[p,q,r,s]
                    op2 += FermionOperator(term, coef)
    return op1, op2, op1 + op2

def calculate_variance_metric(Hferm, obt, tbt_frags):
    """
    Hferm     : electronic Hamiltonian as FermionOperator
    obt       : a one-body-tensor
    tbt_frags : a list of measurable two-body-tensors  

    they should satisfy op(obt) + sum_k op(tbt_frags[k]) == Hferm up to constant

    output    : variance metric value
    """

    # obtain ground state of Hferm
    N      = obt.shape[0]
    psi_GS = get_ground_state(get_sparse_operator(Hferm, N))[1]

    # convert tbt_frags to sparse matrices
    tbt_frags_sparse = list()
    for tbt in tbt_frags:
        _, tbt_fermop, _ = chem_ten2op(np.zeros([N,N]), tbt, N)
        tbt_sparse       = get_sparse_operator(tbt_fermop, N)
        tbt_frags_sparse.append(tbt_sparse) 

    # convert obt to sparse matrix and append to tbt list
    obt_fermop, _, _ = chem_ten2op(obt, np.zeros([N,N,N,N]), N)
    obt_sparse       = get_sparse_operator(obt_fermop, N)

    frag_sparse = [obt_sparse] + tbt_frags_sparse 

    # compute variance of every fragment individually
    fragment_variances = np.zeros(len(frag_sparse), dtype=np.complex128)
    for i, frag in enumerate(frag_sparse):
        fragment_variances[i] = variance(frag, psi_GS)

    # return individual fragment variances and combined variance metric
    return fragment_variances, np.sum(fragment_variances**(1/2))**2

def calculate_variance_metric_memory_efficient(Hferm, obt, tbt_frags):
    """
    same function as calculate_variance_metric, but I compute the sparse matrices one at a time instead of all at once
    to save memory 
    """
    
    # obtain ground state of Hferm
    N = obt.shape[0]
    psi_GS = get_ground_state(get_sparse_operator(Hferm, N))[1]

    # create single list with all fragment tensors, onebody at the 0th index
    frag_tensors = [obt] + tbt_frags

    # iterate through the tensors, and (1) create the corresponding sparse matrix, and (2) compute the variance
    fragment_variances = np.zeros(len(frag_tensors), dtype=np.complex128)
    for i, tensor in enumerate(frag_tensors):
        print(f"fragment number: {i} / {len(frag_tensors)}", end='\r', flush=True)

        if i == 0:
            fermop, _, _ = chem_ten2op(tensor, np.zeros([N,N,N,N]), N)

        else:
            _, fermop, _ = chem_ten2op(np.zeros([N,N]), tensor, N)
            
        sparse_op             = get_sparse_operator(fermop, N)
        fragment_variances[i] = variance(sparse_op, psi_GS)        

    # return individual fragment variances and combined variance metric
    return fragment_variances, np.sum(fragment_variances**(1/2))**2