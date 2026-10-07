# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:19:31 2024

@author: uqalim8
"""
import torch

def CR(A, b, rtol, maxit, reOrtho = True):
    x = torch.zeros_like(b)
    p, r = b, b
    Ap = Avec(A, p)
    
    normr = torch.norm(r)
    normb = normr
    
    normAp = torch.norm(Ap)
    normAb, normAr = normAp, normAp
    if reOrtho:
        AP = Ap.reshape(-1, 1) / normAp
        
    Ar = Ap.clone()
    rAr = torch.dot(r, Ar)
    k = 1
        
    while normr / normb > rtol and k < maxit and rAr > 1e-14:# and normAr / normAb > rtol:
        alpha = rAr / torch.dot(Ap, Ap)
        x = x + alpha * p
        rp1 = r - alpha * Ap
        
        if reOrtho:
            rp1 = rp1 - AP @ (AP.T @ rp1) 
            
        Arp1 = Avec(A, rp1)
        rp1Arp1 = torch.dot(rp1, Arp1)
        beta = rp1Arp1 / rAr
        p = rp1 + beta * p
        Ap = Arp1 + beta * Ap
        
        normAr = torch.norm(Arp1)
        normAp = torch.norm(Ap)
        if reOrtho:
            AP = torch.concat([AP, Ap.reshape(-1, 1) / normAp], dim = 1)
                
        # update
        Ar = Arp1
        rAr = rp1Arp1
        r = rp1
        normr = torch.norm(r)
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
    x, k, relr = CR(H, b, 1e-6, N, True)