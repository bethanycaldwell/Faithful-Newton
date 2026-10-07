# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:19:31 2024

@author: uqalim8
"""
import torch

def CR_NPC(A, b, rtol, maxit):    
    x = torch.zeros_like(b)
    p, r = b, b
    Ap = Avec(A, p)
    
    normr = torch.norm(r)        
    Ar = Ap.clone()
    rAr = torch.dot(r, Ar)
    
    normAr = torch.norm(Ar)
    normAx = 0
    k = 1
    
    # NPC detection
    if rAr <= 0:
        return r * rAr / (normr ** 3), k, "NPC"
    
    while normAr > rtol * normAx and k < maxit:
        alpha = rAr / torch.dot(Ap, Ap)
        x = x + alpha * p
        rp1 = r - alpha * Ap
            
        Arp1 = Avec(A, rp1)
        rp1Arp1 = torch.dot(rp1, Arp1)
        beta = rp1Arp1 / rAr
        p = rp1 + beta * p
        Ap = Arp1 + beta * Ap
                
        # update
        Ar = Arp1
        rAr = rp1Arp1
        r = rp1
        normr = torch.norm(r)
        normAr = torch.norm(Ar)
        normAx = torch.norm(b - r)
        k += 1

        if rAr <= 0:
            return r * rAr / (normr ** 3), k, "NPC"
            
    if k == maxit:
        return x, k, "MAX"
        
    return x, k, "SOL"

def Avec(A, x):
    if callable(A):
        return A(x)
    return torch.mv(A, x)

if __name__ == "__main__":
    N = 100
    H = torch.rand(N, N, dtype = torch.float64)
    H = (H + H.T) / 2
    b = torch.rand(N, dtype = torch.float64)
    x, k, relr = CR_NPC(H, b, 1e-6, N)