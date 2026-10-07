# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:28:54 2024

@author: uqalim8
"""

from .optimizer import Optimizer
from .solvers.faithfulCG import faithfulCG
from .linesearchers.armijo import backwardArmijo
import torch

NEWTON_NC_STATS = {"ite":"g", "inite":"g", "orcs":"g", "time":".2f",
                   "f":".4e", "g_norm":".4e", "alpha":".2e", "acc":".2f"}

class FaithfulNewtonCG(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, beta, skips, inmaxite, 
                 lineMaxite, lineBetaB, lineRho):
        self.info = NEWTON_NC_STATS
        self.beta = beta
        self.inmaxite = inmaxite
        self.lineMaxite = lineMaxite
        self.lineBetaB = lineBetaB
        self.lineite = 0
        self.lineRho = lineRho
        self.skips = skips
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk, self.checks, self.inite, dtype = faithfulCG(self.hk, -self.gk, self.termInner, self.inmaxite, self.skips)
        
        if dtype == "GRD":
            self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), self.xk, 
                                                       self.fk, self.gk, self.alpha0, pk, 
                                                       self.lineBetaB, self.lineRho, self.lineMaxite)
            self.xk += self.alphak * pk
            
        else:
            self.xk += pk
            self.alphak = 1
            
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        self.prev_f = self.fk
        
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.prev_f = self.fk
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((0, 0, 0, 0, float(self.fk), 
                             float(self.gknorm), 0, 0))
        else:
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((self.k, self.inite, self.orcs, 
                               self.toc, float(self.fk), float(self.gknorm), 
                               self.alphak, float(acc)))
            
    def termInner(self, x, xHx):
        #nxt_f = self.fun(self.xk + x, "0")
        #if nxt_f < self.prev_f + self.beta * (torch.dot(self.gk, x) + xHx / 2):
        #    self.prev_f = nxt_f
        #    return True
        #return False
        return self.fun(self.xk + x, "0") < self.fk + self.beta * (torch.dot(self.gk, x))# + xHx / 2)
        
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite + self.checks + self.lineite