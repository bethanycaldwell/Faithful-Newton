# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:54:55 2022

@author: uqalim8
"""
import torch

def CG(A, b, rtol, maxit, reOrtho = True):
    
    x = torch.zeros_like(b)
    p, r = b, b
        
    normb = torch.norm(r)
    normr = normb
    Ap = Avec(A, p)
    pAp = torch.dot(p, Ap)
    if reOrtho:
        R = r.reshape(-1, 1) / normr
        
    k = 1
    while normr / normb > rtol and k < maxit and pAp > 1e-16:
        alpha = (normr ** 2) / pAp
        x = x + alpha * p
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
        pAp = torch.dot(p, Ap)
        k += 1
        
    return x, k, normr / normb

def Avec(A, x):
    if callable(A):
        return A(x)
    return torch.mv(A, x)

if __name__ == "__main__":
    N = 100
    H = torch.rand(N, N, dtype = torch.float64)
    H = (H + H.T) / 2
    b = torch.rand(N, dtype = torch.float64)
    x, k, relr, R, diago, subdiago = CG(H, b, 1e-6, N, True, True)
    