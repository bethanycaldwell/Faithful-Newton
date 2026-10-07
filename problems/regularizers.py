# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:57:40 2022

@author: uqalim8
"""
import torch
TEXT = "{:<20} : {:>20}"

def initReg(reg, lamb):
    
    if reg == "None":
        print(TEXT.format("Regulariser", f"{reg}"))
        return lambda w, v : none_reg(w, v)
    
    if reg == "non-convex":
        print(TEXT.format("Regulariser", f"{reg} , {lamb}"))
        reg = lambda x : non_convex(x, lamb)
        return lambda w, v : fgHv(reg, w, v)
    
    if reg == "2-norm":
        print(TEXT.format("Regulariser", f"{reg} , {lamb}"))
        return lambda x, v : two_norm(x, lamb, v)
    
    if reg == "LASSO":
        print(TEXT.format("Regulariser", f"{reg} , {lamb}"))
        reg = lambda x : LASSO(x, lamb)
        return lambda w, v : fgHv(reg, w, v)
    
    raise ValueError(f"No {reg} regularizer exists")

def two_norm(w, lamb, order):
    w = w.flatten()
    f = lamb * torch.dot(w, w)
    if order == "0":
        return f
    
    g = 2 * lamb * w
    if order == "1":
        return g
    
    if order == "01":
        return f, g
    
    if order == "012":
        return f, g, lambda x : lamb * x 

def non_convex(w, lamb):
    sqw = w ** 2
    return lamb * torch.sum(sqw / (1 + sqw))

def LASSO(w, lamb):
    return lamb * torch.linalg.norm(w, 1)

def none_reg(x, order):
    if order == "012":
        return 0, 0, lambda y : 0
    if order == "0" or order == "1":
        return 0
    if order == "01":
        return 0, 0
    raise ValueError("Order is not regconized")
    
def fgHv(func, w, order = "012"):
    
    with torch.no_grad():
        x = w.clone().requires_grad_(True)
    f = func(x)
    
    if "0" == order:
        return f.detach()
    
    g = torch.autograd.grad(f, x, create_graph = False, retain_graph = True)[0]
    if "1" == order:
        return g.detach()
    
    if "01" == order:
        return f.detach(), g.detach()
    
    if "012" == order:
        g = torch.autograd.grad(f, x, create_graph = True, retain_graph = True)[0]
        Hv = lambda v : torch.autograd.grad((g,), x, v, create_graph = False, retain_graph = True)[0].detach()
        return f.detach(), g.detach(), Hv
    
    raise ValueError("Order is not regconized")
