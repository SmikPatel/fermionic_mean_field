"""
subroutines for implementing low-rank decomposition 

note: this code only works if the g_{pqrs} tensor is real-symmetric and positive-semidefinite when expressed as a matrix

note: there is no main_lowrank.py since the low-rank decompositions take seconds to compute, so I can just compute them
      whenever I need them 
"""

import numpy as np
from openfermion import FermionOperator
from utils_gfro import (
    num_params,
    construct_cartan_tensor,
    PATH
)

def get_onebody_coefficient_matrices(tbt):
    """
    given NxNxNxN two-body-tensor, return NxN matrices L^(k) defining the low-rank fragments
    """
    
    # write the two-body-tensor as a matrix
    N          = tbt.shape[0]
    tbt_matrix = np.reshape(tbt, [N**2, N**2])
    assert np.allclose(tbt_matrix, tbt_matrix.T)

    # diagonalize the matrix
    D, U = np.linalg.eigh(tbt_matrix)

    # sort by absolute value of eigenvalues, largest to smallest
    idx = abs(D).argsort()[::-1]
    D   = D[idx]
    U   = U[:,idx]

    # construct the L^(k) matrices and include them in output if they are large enough in L2 norm
    Ls = list()
    for k in range(len(D)):
        cur_D = np.complex(D[k])
        cur_U = np.reshape(U[:,k], [N,N])
        cur_L = np.real(np.sqrt(cur_D) * cur_U)

        if np.linalg.norm(cur_L) > 1e-8:
            Ls.append(cur_L)

    return Ls

def get_lowrank_fragment_operators(tbt):
    """
    given two-body-tensor, return FermionOperator form of low-rank fragments
    """

    # generate the L^(k) matrices
    N  = tbt.shape[0]
    Ls = get_onebody_coefficient_matrices(tbt)

    # iterate through the L^(k) matrices and compute the corresponding FermionOperator
    Lops = list()

    for L in Ls:

        current_op = FermionOperator()

        for p in range(N):
            for q in range(N):
                term        = ( (p,1), (q,0) )
                coef        = L[p,q]
                current_op += FermionOperator(term, coef)

        Lops.append(current_op * current_op)

    return Lops

def get_lowrank_fragment_tensors(tbt):
    """
    given two-body-tensor, return two-body-tensors of the low-rank fragments
    """

    # generate the L^(k) matrices
    N  = tbt.shape[0] 
    Ls = get_onebody_coefficient_matrices(tbt)

    # iterate through the L^(k) matrices and compute the corresponding two-body-tensor
    Ltbts = list()

    for L in Ls:

        current_tbt = np.zeros([N,N,N,N])

        for p in range(N):
            for q in range(N):
                for r in range(N):
                    for s in range(N):
                        current_tbt[p,q,r,s] += L[p,q] * L[r,s]

        Ltbts.append(current_tbt)

    return Ltbts

def get_full_rank_parameters_for_single_low_rank_fragment(L):
    """
    given low-rank fragment defined by NxN matrix L, return lambdas and orbital rotation that give its form as a full-rank fragment

    note: one could in principle obtain angles that generate the orbital rotation, but these are usually not needed, and require
    effort to obtain
    """
    N       = L.shape[0]
    c, u, p = num_params(N)

    eps, O  = np.linalg.eigh(L)

    lams    = np.zeros(c)

    tally = 0
    for p in range(N):
        for q in range(p, N):
            val          = eps[p] * eps[q]
            assert np.abs(val - np.real(val)) < 1e-12
            lams[tally] += np.real(val)
            if p == q:
                lams[tally] *= 0.5
            tally += 1
    
    return lams, O.T

def get_full_rank_parameters_for_low_rank_fragments(tbt):
    """
    given two-body-tensor, return lambdas and orbital rotations for all low-rank fragments
    """
    N  = tbt.shape[0]
    Ls = get_onebody_coefficient_matrices(tbt)

    lambda_params     = list() 
    orbital_rotations = list()

    for L in Ls:
        cur_lam, cur_rot = get_full_rank_parameters_for_single_low_rank_fragment(L)
        lambda_params.append(cur_lam)
        orbital_rotations.append(cur_rot)

    return lambda_params, orbital_rotations

def two_body_tensor_from_params(lams, O):
    N   = O.shape[0]
    tbt = construct_cartan_tensor(lams, N)

    return np.einsum('ppqq,pa,pb,qc,qd->abcd', tbt, O, O, O, O, optimize=PATH)