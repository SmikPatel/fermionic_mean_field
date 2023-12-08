"""
subroutines for implementing meanfield decompositions
"""

import numpy as np
from utils_gfro import construct_orthogonal
from numpy.random import uniform
from scipy.optimize import minimize

PATH = ['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)]

def num_params(N1, N2):
    N = N1 + N2

    one   = N1*N2*(N1+1)//2
    two   = N2*(N2+1)//2
    three = N*(N-1)//2

    return one, two, three, one+two+three 

def split_params(x, N1, N2):
    c, d, u, p = num_params(N1, N2)

    gam   = x[ : c]
    lam   = x[c : c + d]
    theta = x[c + d : ]

    return gam, lam, theta

def get_fragment(x, S1, S2, N1, N2):
    N               = N1 + N2
    gam, lam, theta = split_params(x, N1, N2)

    O = construct_orthogonal(theta, N)

    tbt = np.zeros([N,N,N,N])

    tally = 0
    for i in S1:
        for j in S1:
            if i >= j:
                for k in S2:
                    tbt[k,k,i,j] += gam[tally]
                    tbt[k,k,j,i] += gam[tally]
                    tbt[i,j,k,k] += gam[tally]
                    tbt[j,i,k,k] += gam[tally]
                    tally        += 1

    tally = 0
    for k in S2:
        for l in S2:
            if k >= l:
                tbt[k,k,l,l] += lam[tally]
                tbt[l,l,k,k] += lam[tally]
                tally        += 1

    return np.einsum('pqrs,pa,qb,rc,sd->abcd', tbt, O, O, O, O, optimize=PATH)

def evaluate_cost_function(x, target_tbt, S1, S2, N1, N2):
    fragment_tbt = get_fragment(x, S1, S2, N1, N2)
    diff         = fragment_tbt - target_tbt

    return np.sum(diff * diff)

def obtain_meanfield_fragment(target_tbt, S1, S2):
    # N and parameter counts
    N1         = len(S1)
    N2         = len(S2)
    N          = N1 + N2
    c, d, u, p = num_params(N1, N2)

    assert N == target_tbt.shape[0]

    # cost function
    def cost(x):
        return evaluate_cost_function(x, target_tbt, S1, S2, N1, N2)

    # initial guess
    x0           = np.zeros(p)
    x0[c + d : ] = uniform(-np.pi/2, np.pi/2, u)

    # options
    options = {
        'maxiter' : 10000,
        'disp'    : False 
    }

    # tolerance
    tol     = 1e-5 
    enum    = N ** 4
    fun_tol = (tol / enum) ** 2

    # optimize
    return minimize(cost, x0, method='BFGS', tol=fun_tol, options=options)