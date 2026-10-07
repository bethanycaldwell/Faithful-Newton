# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:04:59 2024

@author: uqalim8
"""
import torch

def faithfulCR(A, b, term, maxit, skip = 1, T = 1, reOrtho = True):
    k = 1
    p, r = b, b
    Ap = Avec(A, p)
    norm_b = torch.norm(b)
    
    # re-orthogonalization 
    if reOrtho:
        normAp = torch.norm(Ap)
        AP = Ap.reshape(-1, 1) / normAp
        
    Ar = Ap.clone()
    rAr = torch.dot(r, Ar)
    alpha = rAr / torch.dot(Ap, Ap)
    x = alpha * p
    
    norm_Ar = torch.norm(Ar)
    norm_Ab = norm_Ar
    
    # termination condition
    if T == 1: 
        d = 1 # number of times term function is called
        if not term(x, 1):
            return x, d, k, "UNF"
        stored_dir = [(x, 1)]
    
    while k >= T or (norm_Ar / norm_Ab) > 1e-16:        
        rp1 = r - alpha * Ap
        
        # re-orthogonalization 
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
        
        norm_Ar = torch.norm(Ar)
        norm_rm1 = torch.norm(r)
        r = rp1
        k += 1
 
        alpha = rAr / torch.dot(Ap, Ap)
        x = x + alpha * p
        
        # only start monitoring the line-search after T iterations
        if k > T: 
            if not len(stored_dir) % skip:
                d += 1
                if not term(x, (norm_b / norm_rm1) ** 2):
                    return *binary_search(stored_dir, term, d), k, "FTH"
                else:
                    stored_dir = [(x, (norm_b / norm_rm1) ** 2)]
            else:
                stored_dir.append((x, (norm_b / norm_rm1) ** 2))
                
        elif k == T:
            d = 1
            if not term(x, (norm_b / norm_rm1) ** 2):
                return x, d, k, "UNF"
            stored_dir = [(x, (norm_b / norm_rm1) ** 2)]
                        
        # maximum iteration detection 
        if k >= maxit:
            return *binary_search(stored_dir, term, d), k, "FTH"
    return x, 0, k, "SOL"
            
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

def resTerm(x, A, b, tol):
    return torch.norm(b - Avec(A, x)) / torch.norm(b) > tol

if "__main__" == __name__:
    torch.manual_seed(2024)
    N = 100
    MAXIT = 100
    D = torch.rand(N, dtype = torch.float64)
    A = torch.rand(N, N, dtype = torch.float64)
    A = (A.T + A) / 2
    _, eigV = torch.linalg.eigh(A)
    A = eigV.T @ torch.diag(D) @ eigV
    b = torch.rand(N, dtype = torch.float64)
    x, k, relr, dtype = faithfulCR(A, b, lambda x : resTerm(x, A, b, 1e-6), MAXIT, False)
