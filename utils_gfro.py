"""
subroutines for implementing GFRO decomposition
"""

import numpy as np
from scipy.linalg import expm
from numpy.random import uniform
from scipy.optimize import minimize

PATH = ['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)]

def num_params(N):
    c = N * (N + 1) // 2
    u = N * (N - 1) // 2

    return c, u, c + u

def construct_cartan_tensor(coefs, N):

    tbt = np.zeros([N,N,N,N])

    tally = 0
    for p in range(N):
        for q in range(p, N):
            tbt[p,p,q,q] += coefs[tally]
            tbt[q,q,p,p] += coefs[tally]
            tally        += 1

    return tbt

def construct_orthogonal(angles, N):

    X = np.zeros([N,N])

    tally = 0
    for p in range(N):
        for q in range(p+1, N):
            X[p,q] += angles[tally]
            X[q,p] -= angles[tally]
            tally  += 1

    return expm(X)

def get_fragment(x, N):
    c, u, p = num_params(N)

    coefs  = x[ : c]
    angles = x[c : ]

    tbt = construct_cartan_tensor(coefs, N)
    O   = construct_orthogonal(angles, N)

    return np.einsum('ppqq,pa,pb,qc,qd->abcd', tbt, O, O, O, O, optimize=PATH)

def evaluate_cost_function(x, target_tbt, N):
    fragment_tbt = get_fragment(x, N)
    diff         = fragment_tbt - target_tbt

    return np.sum(diff * diff)

def obtain_gfro_fragment(target_tbt):
    # N and parameter counts
    N       = target_tbt.shape[0]
    c, u, p = num_params(N)

    # cost function
    def cost(x):
        return evaluate_cost_function(x, target_tbt, N)

    # initial guess
    x0       = np.zeros(p)
    x0[c : ] = uniform(-np.pi/2, np.pi/2, u)

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