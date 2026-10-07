# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:22:34 2024

@author: uqalim8
"""

import torch

def triCho(alphas, betas):
    """
    Symmetric tridiagonal Cholesky factorization
    
    A = L @ L.T where L is lower bidiagonal matrix
    
    Return:
        diag(D) and sub_diag(D) in 1D tensors
    """
    X = torch.zeros_like(alphas)
    Z = torch.zeros_like(alphas)
    x = None
    signs = betas / torch.abs(betas)
    for n, a in enumerate(alphas):
        if n != 0:
            Z[n] = (betas[n - 1] ** 2) / x 
        x = a - Z[n]
        X[n] = x
    
    X = torch.sqrt(X)
    Z = torch.sqrt(Z)
    assert not (X != X).any()
    assert not (Z != Z).any()
    return X, Z[1:] * signs

def solveLinearTriCho(diago, subdiag, b):
    """
    Let L @ L.T = T be the tridiagonal matrix, where L is the lower bidiagonal 
    Cholesky factorization. 
    
    diago and subdiag are the diagonal and subdiagonal elements of L
    
    solve for x in the linear system, L @ L.T x = b
    """
    subdiag = subdiag / diago[:-1]
    x = torch.zeros_like(b)
    x[0] = b[0]
    l = subdiag.shape[0]
    
    # forward pass
    for n in range(l):
        x[n + 1] = b[n + 1] - subdiag[n] * x[n]
        
    b = x / (diago ** 2)
    x = torch.zeros_like(b)
    x[-1] = b[-1]
    
    # backward pass
    for n in range(l, 0, -1):
        x[n - 1] = b[n - 1] - subdiag[n-1] * x[n]
    
    return x

def forwardPass(diago, subdiag, b):
    """
    Let L @ L.T = T be the tridiagonal matrix, where L is the lower bidiagonal 
    Cholesky factorization. 
    
    diago and subdiag are the diagonal and subdiagonal elements of L
    
    solve for x in the linear system, L x = b
    """
    subdiag = subdiag / diago[:-1]
    x = torch.zeros_like(b)
    x[0] = b[0]
    l = subdiag.shape[0]
    
    # forward pass
    for n in range(l):
        x[n + 1] = b[n + 1] - subdiag[n] * x[n]
        
    return x / diago
    
def invTv(V, alphas, betas, v):
    """
    V is any othonormal matrices
    
    alphas and betas are the diagonal and subdiagonal elements of a symmetric
    tridigonal matrix T.
    
    Solve for x in the linear system, V T^{-1} V.T x = b
    """
    v = V.T @ v
    diago, subdiago = triCho(alphas, betas)
    v = solveLinearTriCho(diago, subdiago, v)
    return V @ v

def formBidiag(diagonal, subdiag):
    D = torch.diag(diagonal)
    SD = torch.diag(subdiag)
    SD = torch.concat([torch.zeros(1, subdiag.shape[0]), SD], dim = 0)
    SD = torch.concat([SD, torch.zeros(subdiag.shape[0] + 1, 1)], dim = 1)
    return D + SD

if "__main__" == __name__:
    N = 100
    diagonal = torch.rand(N, dtype = torch.float64) * 10
    subdiago = torch.rand(N - 1, dtype = torch.float64)
    biDiagon = formBidiag(diagonal, subdiago)
    T = biDiagon @ biDiagon.T
    T_diag = T.diag()
    T_subdiag = T[1:,:-1].diag()
    test_diagonal, test_subdiago = triCho(T_diag, T_subdiag)
    
    assert (torch.abs(test_diagonal - diagonal) < 1e-7).all()
    assert (torch.abs(test_subdiago - subdiago) < 1e-7).all()
    
    b = torch.rand(N, dtype = torch.float64) * 10
    sol = torch.linalg.solve(T, b)
    test_sol = solveLinearTriCho(test_diagonal, test_subdiago, b)
    
    assert (torch.abs(test_sol - sol) < 1e-7).all()
    
    sol = torch.linalg.solve(biDiagon, b)
    test_sol = forwardPass(diagonal, subdiago, b)
    
    assert (torch.abs(test_sol - sol) < 1e-7).all()

    A = torch.rand(N + 100, N + 100, dtype = torch.float64)
    A = A + A.T
    _, vec = torch.linalg.eigh(A)
    
    x = vec[:, :N] @ torch.linalg.inv(T) @ vec[:, :N].T @ torch.ones(N + 100, dtype = torch.float64)
    test_x = invTv(vec[:, :N], T_diag, T_subdiag, torch.ones(N + 100, dtype = torch.float64))
    
    assert (torch.abs(x - test_x) < 1e-7).all()
    