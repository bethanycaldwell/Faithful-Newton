# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 12:33:43 2023

@author: uqalim8
"""

import torch.nn as nn
import torch
from functorch import make_functional

class Wrapper:
    
    def __init__(self):
        self._funcs = {"0" : self.f, "1" : self.g, "2" : self.Hv,
                       "01" : self.fg, "02" : self.fHv, "12" : self.gHv,
                       "012": self.fgHv}
        
    def f(self, w):
        raise NotImplementedError
    
    def g(self, w):
        raise NotImplementedError
    
    def Hv(self, w):
        raise NotImplementedError
    
    def fg(self, w):
        raise NotImplementedError
    
    def fgHv(self, w):
        raise NotImplementedError
    
    def gHv(self, w):
        raise NotImplementedError
        
    def fHv(self, w):
        raise NotImplementedError

    def __call__(self, x, order):
        return self._funcs[order](x)
    
class nnWrapper(Wrapper):
    
    def __init__(self, func, loss, reg, X, Y):
        super().__init__()
        self.func, self.loss = func, loss
        self.X, self.Y = X, Y
        self.reg = reg
        self.size = nn.utils.parameters_to_vector(func.parameters()).shape[-1]
    
    def _toModule_toFunctional(self, w):
        w = w.clone()
        if w.requires_grad:
            w = w.detach().requires_grad_(True)
        else:
            w = w.requires_grad_(True)
        nn.utils.vector_to_parameters(w, self.func.parameters())
        return make_functional(self.func, disable_autograd_tracking = False)
    
    def f(self, x):
        functional, w = self._toModule_toFunctional(x)
        with torch.no_grad():
            return self.loss(functional(w, self.X), self.Y)
            
    def g(self, x):
        functional, w = self._toModule_toFunctional(x)
        val = self.loss(functional(w, self.X), self.Y)
        g = torch.autograd.grad(val, w)
        g = nn.utils.parameters_to_vector(g)
        return g.detach()
    
    def fg(self, x):
        functional, w = self._toModule_toFunctional(x)
        val = self.loss(functional(w, self.X), self.Y)
        g = torch.autograd.grad(val, w)
        g = nn.utils.parameters_to_vector(g)
        return val.detach(), g.detach()
    
    def fgHv(self, x):
        functional, w = self._toModule_toFunctional(x)
        val = self.loss(functional(w, self.X), self.Y)
        g = torch.autograd.grad(val, w, create_graph = True)
        g = nn.utils.parameters_to_vector(g)
        Hv = lambda v : nn.utils.parameters_to_vector(
            torch.autograd.grad(g, w, grad_outputs = v, create_graph = False, retain_graph = True)
            ).detach()
        return val.detach(), g.detach(), Hv
    
    def __call__(self, x, order):
    
        if order == "0":
            return self.f(x) + self.reg(x, "0")
        
        if order == "1":
            return self.g(x) + self.reg(x, "1")
        
        if order == "01":
            f, g = self.fg(x)
            reg_f, reg_g = self.reg(x, order)
            return f + reg_f, g + reg_g
        
        if order == "012":
            f, g, Hv = self.fgHv(x)
            reg_f, reg_g, reg_Hv = self.reg(x, order)
            return f + reg_f, g + reg_g, lambda x : Hv(x) + reg_Hv(x)
        
        raise ValueError("Order is not regconized")
        
class funcWrapper(Wrapper):

    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def _gradIt(self, w):
        if w.requires_grad:
            return w.detach().requires_grad_(True)
        else:
            return w.requires_grad_(True)
        
    def f(self, w):
        with torch.no_grad():
            return self.func(w)
            
    def g(self, w):
        w = self._gradIt(w)
        f = self.func(w)
        g = torch.autograd.grad(f, w)[0]
        f.detach()
        return g.detach()
    
    def fg(self, w):
        w = self._gradIt(w)
        f = self.func(w)
        g = torch.autograd.grad(f, w)[0]
        return f.detach(), g.detach()
    
    def fgHv(self, w):
        w = self._gradIt(w)
        f = self.func(w)
        g = torch.autograd.grad(f, w, create_graph = True)[0]
        Hv = lambda v : torch.autograd.grad((g,), w, v, create_graph = False, 
                                            retain_graph = True)[0].detach()
        return f.detach(), g.detach(), Hv
    
if __name__ == "__main__":
    from derivativeTest import derivativeTest
    from neural_network import FFN
    
    def none_reg(x, order):
        if order == "012":
            return 0, 0, lambda y : 0
        if order == "0" or order == "1":
            return 0
        if order == "01":
            return 0, 0
        raise ValueError("Order is not regconized")
    
    ffn = FFN(10, 5)
    X = torch.rand(100, 10, dtype = torch.float64)
    Y = torch.rand(100, 5, dtype = torch.float64)
    loss = nn.MSELoss()
    f = nnWrapper(ffn, loss, none_reg, X, Y)
    x0 = nn.utils.parameters_to_vector(ffn.parameters()).to(torch.float64)
    derivativeTest(lambda x : f(x, "012"), x0)