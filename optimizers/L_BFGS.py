# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:02:16 2024

@author: uqalim8
"""

import torch
from .optimizer import Optimizer
from .linesearchers.strongWolfe import lineSearchWolfeStrong

L_BFGS_STATS = {"ite":"g", "orcs":"g", "time":".2f", "f":".4e", "g_norm":".4e", "iteLS":"g", 
                "alpha":".2e", "acc":".2f"}

class L_BFGS(Optimizer):

    def __init__(self, fun, x0, alpha0, gradtol, m, maxite, maxorcs, lineMaxite):
        self.info = L_BFGS_STATS
        self.m = m
        self.lineMaxite = lineMaxite
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def _twoloop(self, w):
        k = self.s.shape[0]
        alpha, rho = torch.zeros_like(self.s), torch.zeros_like(self.s)
        for i in range(k):
            rho[i] = 1/torch.dot(self.s[i], self.y[i])
            alpha[i] = rho[i] * torch.dot(self.s[i], w)
            w = w - alpha[i] * self.y[i]
            
        w = ((torch.dot(self.s[0], self.y[0])) / torch.dot(self.y[0], self.y[0])) * w
        for i in range(k - 1, -1, -1):
            beta = rho[i] * torch.dot(self.y[i], w)
            w = w + (alpha[i] - beta) * self.s[i]

        return w
    
    def step(self):
        
        if not self.k:
            pk = -self.gk
        else:
            pk = self._twoloop(-self.gk).detach()
        
        self.alpha, self.lineite, self.lineorcs = lineSearchWolfeStrong(lambda x : self.fun(x, "01"), self.xk, pk, 
                                                         self.fk, self.gk, self.alpha0, 1e5, 1e-4, 0.5, self.lineMaxite)
        xkp1 = self.xk + self.alpha * pk
        self.fk, gkp1 = self.fun(xkp1, "01")

        # kill small alpha and terminate
        if self.alpha == 0:
            self.orcs = self.maxorcs
            self.lineorcs = 0

        if self.k and self.s.shape[0] >= self.m:
            self.s = self.s[:-1]
            self.y = self.y[:-1]
            
        temps = xkp1 - self.xk
        tempy = gkp1 - self.gk     
        
        if not self.k:
            self.s = temps.reshape(1, -1)
            self.y = tempy.reshape(1, -1)
        else:
            self.s = torch.cat([temps.reshape(1, -1), self.s], dim = 0)
            self.y = torch.cat([tempy.reshape(1, -1), self.y], dim = 0)
            
        self.gk = gkp1
        self.xk = xkp1

    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk = self.fun(self.xk, "01")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, 0, float(self.fk), 
                             float(self.gknorm), 0, 0, acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((self.k, self.orcs, self.toc, float(self.fk), 
                            float(self.gknorm), self.lineite, float(self.alpha), acc))  
            
    def oracleCalls(self):
        self.orcs += 2 + self.lineorcs
