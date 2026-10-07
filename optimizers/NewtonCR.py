# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:19:00 2024

@author: uqalim8
"""

import torch 
from .optimizer import Optimizer
from .linesearchers.armijo import backwardArmijo
from .solvers.CR import CR

NEWTON_STATS = {"ite":"g", "inite":"g", "orcs":"g", "time":".2f", 
                "f":".4e", "g_norm":".4e", "alpha":".2e", "acc":".2f"}

class NewtonCR(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, restol, inmaxite, 
                 lineMaxite, lineBetaB, lineRho):
        self.info = NEWTON_STATS
        self.restol = restol
        self.inmaxite = inmaxite
        self.lineMaxite = lineMaxite
        self.lineBetaB = lineBetaB
        self.lineRho = lineRho
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk, self.inite, rtol = CR(self.hk, -self.gk, self.restol, self.inmaxite)
        self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), 
                                                   self.xk, self.fk, self.gk, self.alpha0, pk, 
                                                   self.lineBetaB, self.lineRho, self.lineMaxite)
        self.xk += self.alphak * pk
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        self.gknorm = torch.linalg.norm(self.gk, torch.inf)
    
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((0, 0, 0, 0, float(self.fk), float(self.gknorm), 0, acc))
        else:
            self.recording((self.k, self.inite, self.orcs, self.toc, 
                               float(self.fk), float(self.gknorm), self.alphak, acc))
        
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite + self.lineite
        
