# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:00:56 2024

@author: uqalim8
"""
from .optimizer import Optimizer
from .solvers.MINRES import myMINRES
from .linesearchers.armijo import backForwardArmijo, backwardArmijo
import torch

NEWTON_NC_STATS = {"ite":"g", "inite":"g", "dtype":"", "orcs":"g", "time":".2f",
                   "f":".4e", "g_norm":".4e", "alpha":".2e", "acc":".2f"}

class NewtonMR_NC(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, restol, inmaxite, 
                 lineMaxite, lineBetaB, lineRho, lineBetaFB):
        self.info = NEWTON_NC_STATS
        self.restol = restol
        self.inmaxite = inmaxite
        self.lineMaxite = lineMaxite
        self.lineBetaB = lineBetaB
        self.lineRho = lineRho
        self.lineBetaFB = lineBetaFB
        self.alpha_npc = 1
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk, self.relr, self.inite, r, self.dtype = myMINRES(self.hk, -self.gk, rtol = self.restol, maxit = self.inmaxite)
        if self.dtype == "Sol" or self.dtype == "MAX":
            self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), 
                                                       self.xk, self.fk, self.gk, self.alpha0, pk, 
                                                       self.lineBetaB, self.lineRho, self.lineMaxite)
        else:
            self.alphak, self.lineite = backForwardArmijo(lambda x : self.fun(x, "0"), 
                                                          self.xk, self.fk, self.gk, self.alpha_npc, r, 
                                                          self.lineBetaFB, self.lineRho, self.lineMaxite)
            self.alpha_npc = self.alphak
            pk = r
        
        self.xk += self.alphak * pk
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((0, 0, "None", 0, 0, float(self.fk), 
                             float(self.gknorm), 0, 0))
        else:
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((self.k, self.inite, self.dtype, self.orcs, 
                               self.toc, float(self.fk), float(self.gknorm), 
                               self.alphak, float(acc)))
        
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite + self.lineite