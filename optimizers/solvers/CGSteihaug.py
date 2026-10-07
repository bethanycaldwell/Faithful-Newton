# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:44:53 2024

@author: uqalim8
"""
import torch

def CGSteihaug(H, g, delta, rtol, maxite):
    
    z = torch.zeros_like(g)
    # if torch.norm(g) < tol:
    #     return z, "||g||<tol", 1, 0
    
    norm_g = torch.norm(g)
    j = 0
    d, r = -g.clone(), g.clone()
    while j <= maxite:
        Bd = Avec(H, d)
        dBd = torch.dot(d, Bd)
        j += 1
        if dBd <= 0:
            dz = torch.dot(d, z)
            norm_d, norm_z = torch.norm(d), torch.norm(z)
            numerator = - dz + torch.sqrt(dz**2  - norm_d**2 * (norm_z**2 - delta**2))
            tau = numerator / norm_d**2
            p = z + tau * d
            m0_mk = - torch.dot(g, p) - torch.dot(p, Avec(H, p)) / 2
            return p, "NC", m0_mk, j
        
        norm_r = torch.dot(r, r)
        alpha = norm_r / dBd
        zp1 = z + alpha * d
        if torch.norm(zp1) >= delta:
            dz = torch.dot(d, z)
            norm_d, norm_z = torch.norm(d), torch.norm(z)
            numerator = - dz + torch.sqrt(dz**2  - norm_d**2 * (norm_z**2 - delta**2))
            tau = numerator / norm_d**2
            p = z + tau * d
            m0_mk = - torch.dot(g, p) - torch.dot(p, Avec(H, p)) / 2
            return p, "SOL,=", m0_mk, j

        z = zp1
        r = r + alpha * Bd
        if torch.norm(r) / norm_g < rtol:
            p = z
            m0_mk = - torch.dot(g, p) - torch.dot(p, Avec(H, p)) / 2
            return p, "SOL,<", m0_mk, j
        
        norm_rp1 = torch.dot(r, r)
        beta = norm_rp1 / norm_r
        d = -r + beta * d
        norm_r = norm_rp1
    
    p = z
    m0_mk = - torch.dot(g, p) - torch.dot(p, Avec(H, p)) / 2

    return p, "MAX,<", m0_mk, j 

def Avec(A, x):
    if callable(A):
        return A(x)
    return torch.mv(A, x)