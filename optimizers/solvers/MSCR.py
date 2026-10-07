# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:19:31 2024

@author: uqalim8
"""
import torch

def OaMSNfo(A, b, y, fun, lamb, sig, lazy, CRmaxit, maxIt):
    
    failedCheck = False
    total_k = 0
    xk = None
    for i in range(maxIt):
        Ap = lambda x : Avec(A, x) + lamb * x
        xk, k = CR(Ap, b, xk, lamb, sig, CRmaxit)
        total_k += 1 + k
        newxk = y + xk
        fk, gk = fun(newxk, "01")
        if torch.norm(newxk - (y - gk / lamb)) <= sig * torch.norm(newxk - y):
            if lazy or failedCheck:
                #print(torch.norm(newxk))
                return newxk, lamb, total_k
            else:
                lamb /= 2
        else:
            failedCheck = True
            lamb *= 2
        
        
def CR(A, b, u, lamb, sig, maxit, reOrtho = False):
    if u is None:
        u = torch.zeros_like(b)
        p, r = b, b
        Ap = Avec(A, p)
    else:
        p = b - Avec(A, u)
        r = p.clone()
        Ap = Avec(A, p)

    normr = torch.norm(r)
    normu = torch.norm(u)
    normAp = torch.norm(Ap)
    if reOrtho:
        AP = Ap.reshape(-1, 1) / normAp
        
    Ar = Ap.clone()
    rAr = torch.dot(r, Ar)
    k = 1
    while normr > (sig * lamb) * normu / 2 and k < maxit:
        alpha = rAr / torch.dot(Ap, Ap)
        u = u + alpha * p
        normu = torch.norm(u)
        rp1 = r - alpha * Ap
        
        if reOrtho:
            rp1 = rp1 - AP @ (AP.T @ rp1) 
            
        Arp1 = Avec(A, rp1)
        rp1Arp1 = torch.dot(rp1, Arp1)
        beta = rp1Arp1 / rAr
        p = rp1 + beta * p
        Ap = Arp1 + beta * Ap
        
        if reOrtho:
            normAp = torch.norm(Ap)
            AP = torch.concat([AP, Ap.reshape(-1, 1) / normAp], dim = 1)
                
        # update
        Ar = Arp1
        rAr = rp1Arp1
        r = rp1
        normr = torch.norm(r)
        k += 1
    return u, k

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