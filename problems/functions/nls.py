# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:53:55 2024

@author: uqalim8
"""

import torch

def nls(X, y, w, order = "012"):      
    n, d = X.shape
    f, g, H = None, None, None
    model_f, model_g, model_H = logisticRegression(X, w, order)
    if "0" in order:
       f = torch.sum((model_f - y) ** 2) / n
    
    if "1" in order:    
        g = 2 * torch.mv(X.T, model_g * (model_f - y)) / n
    
    if "2" in order:
        model_H = 2 * ((model_f - y) * model_H + model_g ** 2) / n
        H = lambda v : hess_vec_product(X, model_H, v)
    
    return f, g, H

def hess_vec_product(X, H, v):
    Xv = torch.mv(X, v)
    return torch.mv(X.T, H * Xv)

def logisticRegression(X, w, order):
    n, d = X.shape
    t = torch.mv(X, w)
    M = torch.max(torch.zeros_like(t),t)
    a = torch.exp(-M) + torch.exp(t-M)
    s = torch.exp(t-M)/a
    g, H = None, None
    
    if "1" in order:    
        g = s*(1-s)
    
    if "2" in order:    
        H = s*(1-s)*(1-2*s)

    return s, g, H

if __name__ == "__main__":
    from derivativeTest import derivativeTest
    
    n, d = 1000, 50
    A = torch.randn((n, d), dtype = torch.float64)
    b = torch.randint(0, 2, (n,))
    fun = lambda x : nls(A, b, x, "012")
    derivativeTest(fun, torch.ones(50, dtype = torch.float64))