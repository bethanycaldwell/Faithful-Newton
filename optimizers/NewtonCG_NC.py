# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:09:33 2024

@author: uqalim8
"""

from .optimizer import Optimizer
from .solvers.CappedCG import CappedCG
from .linesearchers.cubic import dampedNewtonCGLinesearch
import torch

NEWTON_NC_STATS = {"ite":"g", "inite":"g", "dtype":"", "orcs":"g", "time":".2f",
                   "f":".4e", "g_norm":".4e", "alpha":".2e", "acc":".4e"}

class NewtonCG_NC(Optimizer):
    
    # Without second order optimality 
    # Simplified, i.e. without minimum eigenvalue oracle
    # Without forward linesearch
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, restol, inmaxite,
                 lineMaxite, lineBeta, lineRho, epsilon):
        self.info = NEWTON_NC_STATS
        self.restol = restol
        self.inmaxite = inmaxite
        self.lineMaxite = lineMaxite
        self.lineBeta = lineBeta
        self.lineRho = lineRho
        self.epsilon = epsilon
        self.alpha0 = alpha0
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
    
    def step(self):
        pk, self.dtype, self.inite, pHp, _ = CappedCG(self.hk, -self.gk, self.restol, self.epsilon, self.inmaxite)
        normpk = torch.linalg.norm(pk, 2)**3
        if self.dtype == "NC":
            pk = - torch.sign(torch.dot(pk, self.gk)) * abs(pHp) * pk / normpk
        self.alphak, self.lineite = dampedNewtonCGLinesearch(lambda x : self.fun(x, "0"), self.xk, self.fk, self.alpha0, pk, 
                                                             normpk, self.lineBeta, self.lineRho, self.lineMaxite)

        self.xk += self.alphak * pk
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, "None", 0, 0, float(self.fk), 
                             float(self.gknorm), 0, acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((self.k, self.inite, self.dtype, self.orcs, 
                               self.toc, float(self.fk), float(self.gknorm), self.alphak, acc))
            
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite + self.lineite