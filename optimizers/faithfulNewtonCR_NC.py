# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:28:54 2024

@author: uqalim8
"""

from .optimizer import Optimizer
from .solvers.faithfulCR_NPC import faithfulCR_NPC
from .linesearchers.armijo import backwardArmijo, backForwardArmijo
import torch

NEWTON_NC_STATS = {"ite":"g", "inite":"g", "relRes":".4e", "orcs":"g", "time":".2f",
                   "f":".4e", "g_norm":".4e", "alpha":".2e", "acc":".2f"}

class FaithfulNewtonCR_NC(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, beta, skips, inmaxite, 
                 lineMaxite, lineBetaB, lineRho):
        self.info = NEWTON_NC_STATS
        self.beta = beta
        self.inmaxite = inmaxite
        self.lineMaxite = lineMaxite
        self.lineBetaB = lineBetaB
        self.lineRho = lineRho
        self.lineite = 0
        self.skips = skips
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        self.pk, self.checks, self.inite, self.dtype = faithfulCR_NPC(self.hk, -self.gk, self.termInner, self.inmaxite, self.skips)
        
        if self.dtype == "GRD":
            self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), self.xk, 
                                                       self.fk, self.gk, self.alpha0, self.pk, 
                                                       self.lineBetaB, self.lineRho, self.lineMaxite)
            self.xk += self.alphak * self.pk
            
        elif self.dtype == "NPC":
            self.alphak, self.lineite = backForwardArmijo(lambda x : self.fun(x, "0"), self.xk, 
                                                       self.fk, self.gk, self.alpha0, self.pk, 
                                                       self.lineBetaB, self.lineRho, self.lineMaxite)
            self.xk += self.alphak * self.pk
            
        else:
            self.xk += self.pk
            self.alphak = 1
            
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((0, 0, 0, 0, 0, float(self.fk), 
                             float(self.gknorm), 0, 0))
        else:
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((self.k, self.inite, float(self.res), self.orcs, 
                               self.toc, float(self.fk), float(self.gknorm), 
                               self.alphak, float(acc)))
            
    def termInner(self, x, inv_relres):
        return self.fun(self.xk + x, "0") < self.fk + self.beta * inv_relres * torch.dot(self.gk, x)
        
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite + self.checks + self.lineite
