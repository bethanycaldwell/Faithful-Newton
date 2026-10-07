# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:23:47 2024

@author: uqalim8
"""
import torch

def softmax(X, Y, w, order, reg):  
    """
    All vectors are column vectors.
    INPUTS:
        X: a nxd data matrix.
        Y: a nxC label matrix, C = total class number - 1
        w: starting point
        HProp: porposion of Hessian perturbation
        arg: output control
        reg: a function handle of regulizer function that returns f,g,Hv
    OUTPUTS:
        f: objective function value
        g: gradient of objective function
        Hv: a Hessian-vector product function handle of any column vector v
    """
    n, d = X.shape
    C = int(len(w)/d)
    w = w.reshape(d*C,1)
    W = w.reshape(C,d).T
    XW = torch.mm(X,W)
    large_vals = torch.max(XW,axis = 1)[0]
    large_vals = torch.clamp(large_vals, min=0)
    XW_trick = XW - large_vals.repeat(C, 1).T
    XW_1_trick = torch.cat((-large_vals.reshape(-1,1), XW_trick), 1)
    sum_exp_trick = torch.sum(torch.exp(XW_1_trick), axis = 1)
    log_sum_exp_trick = large_vals + torch.log(sum_exp_trick)
    f = torch.sum(log_sum_exp_trick)/n - torch.sum(torch.sum(XW*Y,axis=1))/n
    
    if order == '0':        
        return f + reg(w, "0")
    
    inv_sum_exp = 1./sum_exp_trick
    inv_sum_exp = inv_sum_exp.repeat(C, 1).T
    S = inv_sum_exp*torch.exp(XW_trick)
    g = torch.mm(X.T, S-Y)/n 
    g = g.T.flatten()
    
    if order == '1':
        return g + reg(w, "1")
    
    if order == '01':
        reg_f, reg_g = reg(w, "01")
        return f + reg_f, g + reg_g
    
    if order == "012":
        reg_f, reg_g, reg_Hv = reg(w, "012")
        return f + reg_f, g + reg_g, lambda v: hessvec(X, S, n, v, d, C) + reg_Hv(v)
    
    raise ValueError("Order input is not regconized")
 
def hessvec(X, S, n, v, d, C):
    v = v.reshape(len(v),1)
    V = v.reshape(C, d).T #[d x C]
    A = torch.mm(X,V) #[n x C]
    AS = torch.sum(A*S, axis=1)
    rep = AS.repeat(C, 1).T #A.dot(B)*e*e.T
    XVd1W = A*S - S*rep #[n x C]
    Hv = torch.mm(X.T, XVd1W)/n #[d x C]
    Hv = Hv.T.flatten() #[d*C, ] #[d*C, ]
    return Hv

if __name__ == "__main__":
    from derivativeTest import derivativeTest
    n, d = 1000, 50
    A = torch.randn((n, d), dtype = torch.float64)
    b = torch.randint(0, 2, (n * 10,))
    b = b.reshape(n, -1)
    fun = lambda x : softmax(A, b, x, reg=None, arg = "012")
    derivativeTest(fun, torch.ones(50 * 10, dtype = torch.float64))