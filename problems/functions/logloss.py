# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:54:53 2022

@author: uqalim8
"""
import torch

def logisticFun(x, A, b, reg, order_deriv = "012", H_matrix = False):
    Ax = torch.mv(A, x)
    c = torch.maximum(Ax, torch.zeros_like(Ax))
    expc = torch.exp(-c)
    expAx = torch.exp(Ax - c)
    f = torch.sum(c + torch.log(expc + expAx) - b * Ax) + 0.5 * reg * torch.linalg.norm(x)**2
    
    if "0" == order_deriv:
        return f
    
    t = expAx/(expc + expAx)
    if "01" == order_deriv:
        g = torch.sum((t - b).reshape(-1, 1) * A, axis = 0) + reg * x
        return f, g
    
    g = torch.sum((t - b).reshape(-1, 1) * A, axis = 0) + reg * x
    
    if H_matrix:
        H = A.T @ ((t * (1 - t)).reshape(-1, 1) * A) + reg * torch.eye(len(x))
        
    else:
        H = lambda v : A.T @ ((t * (1 - t)).reshape(-1, 1) * A @ v) + reg * v
    
    return f, g, H

def logisticModel(A, w):
    expo = - torch.mv(A, w)
    c = torch.maximum(expo, torch.zeros_like(expo, dtype = torch.float64))
    expc = torch.exp(-c)
    de = expc + torch.exp(- c + expo)
    return expc / de

if __name__ == "__main__":
    from derivativeTest import derivativeTest
    
    n, d = 1000, 50
    A = torch.randn((n, d), dtype = torch.float64)
    b = torch.randint(0, 2, (n,))
    fun = lambda x : logisticFun(x, A, b, 1, "012")
    derivativeTest(fun, torch.ones(d, dtype = torch.float64))