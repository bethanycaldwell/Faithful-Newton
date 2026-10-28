# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:23:47 2024

@author: uqalim8
"""
import torch

def softmax_cubic(X, Y, w, order, reg, M):  
    """
    Using f as the softmax function, this function computes the cubic regularized softmax function value, gradient, and Hessian-vector product.
    All vectors are column vectors.
    INPUTS:
        X: a nxd data matrix.
        Y: a nxC label matrix, C = total class number - 1
        w: starting point
        M: cubic regularization parameter
        order: output control
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
    
    inv_sum_exp = 1./sum_exp_trick
    inv_sum_exp = inv_sum_exp.repeat(C, 1).T
    S = inv_sum_exp*torch.exp(XW_trick)
    g = torch.mm(X.T, S-Y)/n
    g = g.T.flatten()
    w = torch.flatten(w)
    reg_f, reg_g, reg_Hv = reg(w, "012")
    if order == '0' and reg_f == 0:  
        fc = f + torch.dot(g, w) + torch.dot(hessvec(X, S, n, w, d, C), w) / 2 + M * (torch.norm(w) ** 3) / 6  
        return fc
    
    if order == '1' and reg_f == 0:
        gc = g + hessvec(X, S, n, w, d, C) + M * (torch.norm(w) ** 2) * w / 2
        return gc
    
    if order == '01' and reg_f == 0:
        fc = f + torch.dot(g, w) + torch.dot(hessvec(X, S, n, w, d, C), w) / 2 + M * (torch.norm(w) ** 3) / 6
        gc = g + hessvec(X, S, n, w, d, C) + M * (torch.norm(w) ** 2) * w / 2
        return fc, gc
    
    if order == "012" and reg_f == 0:
        fc = f + torch.dot(g, w) + torch.dot(hessvec(X, S, n, w, d, C), w) / 2 + M * (torch.norm(w) ** 3) / 6
        gc = g + hessvec(X, S, n, w, d, C) + M * (torch.norm(w) ** 2) * w / 2
        return fc, gc, lambda v:hessvec(X, S, n, v, d, C) + M * torch.norm(v) * v


    if order == '0' and reg_f != 0:  
        fc = f + reg_f + torch.dot(g, w) + torch.dot(reg_g,w) + torch.dot(hessvec(X, S, n, w, d, C) + reg_Hv(w), w) / 2 + M * (torch.norm(w) ** 3) / 6   
        return fc
    
    if order == '1' and reg_f != 0:
        gc = g + reg_f + hessvec(X, S, n, w, d, C) + reg_Hv(w) + M * (torch.norm(w) ** 2) * w / 2
        return gc
    
    if order == '01' and reg_f != 0:
        fc = f + reg_f + torch.dot(g, w) + torch.dot(reg_g,w) + torch.dot(hessvec(X, S, n, w, d, C) + reg_Hv(w), w) / 2 + M * (torch.norm(w) ** 3) / 6
        gc = g + reg_g + hessvec(X, S, n, w, d, C) + reg_Hv(w) + M * (torch.norm(w) ** 2) * w / 2
        return fc, gc
        
    if order == "012" and reg_f != 0:
        fc = f + reg_f + torch.dot(g, w) + torch.dot(reg_g,w) + torch.dot(hessvec(X, S, n, w, d, C) + reg_Hv(w), w) / 2 + M * (torch.norm(w) ** 3) / 6
        gc = g + reg_g + hessvec(X, S, n, w, d, C) + reg_Hv(w) + M * (torch.norm(w) ** 2) * w / 2
        return fc, gc, lambda v:hessvec(X, S, n, v, d, C) + reg_Hv(v) + M * torch.norm(v) * v
    
    raise ValueError("Order input is not regconized")
 
def hessvec(X, S, n, v, d, C):
    """
    Parameters:
    X (torch.Tensor): The data matrix of shape (n, d).
    S (torch.Tensor): The softmax probabilities of shape (n, C).
    n (int): The number of samples.
    v (torch.Tensor): The vector for which to compute the Hessian-vector product.
    d (int): The number of features.
    C (int): The number of classes.

    Returns:
    torch.Tensor: The Hessian-vector product of shape (d*C,).
    """
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
    fun = lambda x : softmax_cubic(A, b, x, reg=None, arg = "012", M = 1e3)
    derivativeTest(fun, torch.ones(50 * 10, dtype = torch.float64))