# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:22:15 2024

@author: uqalim8
"""

import torch

def faithfulCG(A, b, term, maxit, skip = 1, reOrtho = True):
    k = 1
    p, r = b, b
        
    normb = torch.norm(r)
    normr = normb
    Ap = Avec(A, p)

    if reOrtho:
        R = r.reshape(-1, 1) / normr
        
    pAp = torch.dot(p, Ap)
    alpha = (normr ** 2) / pAp
    x = alpha * p
       
    d = 1 # number of times term is called
    
    xHx = torch.dot(x, alpha * Ap)
    if not term(x, None):
        if alpha < 1:
            return x / 2, d, k, "GRD"
        return b, d, k, "GRD"
    
    stored_dir = [(x, None)]
    #skip = 1
    while True:
        r = r - alpha * Ap
        if reOrtho:
            r = r - R @ (R.T @ r)
            normrkp1 = torch.norm(r)
            R = torch.concat([R, r.reshape(-1, 1) / normrkp1], dim = 1)
        else:
            normrkp1 = torch.norm(r)
        
        beta = (normrkp1 / normr) ** 2
        p = r + beta * p
        Ap = Avec(A, p)
        normr = normrkp1
        k += 1
        
        pAp = torch.dot(p, Ap)
        alpha = (normr ** 2) / pAp
        x = x + alpha * p

        #xHx = torch.dot(x, b - (r - alpha * Ap))
        # only check for termination condition every skip_check times
        if not len(stored_dir) % skip and not term(x, None):
            return *binary_search(stored_dir, term, d), k, "TER"
        
        # clear stored update directions
        if not len(stored_dir) % skip:
            d += 1
            stored_dir = [(x, None)]
            #skip = min(skip * 2, max_skip)
        else:
            stored_dir.append((x, None))
        
        if k >= maxit:
            return *binary_search(stored_dir, term, d), k, "MAX"
            
def binary_search(xs, term, d):
    t = len(xs)
    if t == 1:
        return xs[0][0], d 
    t, d = t // 2, d + 1
    if term(xs[t][0], xs[t][1]):
        return binary_search(xs[t:], term, d)
    return binary_search(xs[:t], term, d)
        
def Avec(A, x):
    if callable(A):
        return A(x)
    return torch.mv(A, x)

def termination(A, b, x, k, rtol = 1e-6):
    return torch.norm(Avec(A, x) - b) / torch.norm(b) > rtol

if __name__ == "__main__":
    torch.manual_seed(1234)
    N = 100
    MAXIT = 100
    D = torch.randn(N, dtype = torch.float64)
    A = torch.rand(N, N, dtype = torch.float64)
    A = (A.T + A) / 2
    _, eigV = torch.linalg.eigh(A)
    A = eigV.T @ torch.diag(D) @ eigV
    b = torch.rand(N, dtype = torch.float64)
    x, k, relr = faithfulCG(A, b, lambda x, k : termination(A, b, x, k), MAXIT, False)