# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:31:13 2022

@author: uqalim8
"""

from linesearch import (backwardArmijo, 
                        backForwardArmijo,
                        backForwardArmijo_mod, 
                        dampedNewtonCGLinesearch, 
                        dampedNewtonCGbackForwardLS, 
                        lineSearchWolfeStrong)
import torch, time
from CG import CG, CappedCG, CGSteihaug
from MINRES import myMINRES
from hyperparameters import cTYPE, cCUDA

GD_STATS = {"ite":"g", "orcs":"g", "time":".2f", "f":".4e", 
            "g_norm":".4e", "alpha":".2e", "acc":".2f"}

SGD_STATS = {"ite":"g", "orcs":"g", "time":".2f", "f":".4e", 
            "g_norm":".4e", "acc":".2f"}



NEWTON_TR_STATS = {"ite":"g", "inite":"g", "dtype":"", "orcs":"g", "time":".2f",
                   "f":".4e", "g_norm":".4e", "delta":".2e", "acc":".2f"}

class linesearchGD(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, lineMaxite, lineBetaB, lineRho):
        self.info = GD_STATS
        self.lineMaxite = lineMaxite
        self.lineBetaB = lineBetaB
        self.lineRho = lineRho
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk = -self.gk
        self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), self.xk, self.fk, self.gk, self.alpha0, pk,
                                     self.lineBetaB, self.lineRho, self.lineMaxite)
        self.xk += self.alphak * pk
        self.fk, self.gk = self.fun(self.xk, "01")
        self.gknorm = torch.linalg.norm(self.gk, 2)

    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk = self.fun(self.xk, "01")
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, 0, float(self.fk), float(self.gknorm), 0, acc))
        else:
            self.recording((self.k, self.orcs, self.toc, 
                               float(self.fk), float(self.gknorm), self.alphak, acc))
        
    def oracleCalls(self):
        self.orcs += 2 + self.lineite
        
class MiniBatchSGD(Optimizer):
    
    def __init__(self, fun, x0, gradtol, maxite, maxorcs, mini, alpha = 0.001):
        self.info = SGD_STATS
        self.mini = mini
        super().__init__(fun, x0, alpha, gradtol, maxite, maxorcs)
    
    def step(self):
        self.gk = self.fun(self.xk, "1")
        self.fk = self.fun(self.xk, "f")
        self.xk -= self.alpha0 * self.gk
        
    def recordStats(self, acc):
        if self.k == 0:
            self.gk = self.fun(self.xk, "1")
            self.fk = self.fun(self.xk, "f")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, 0, float(self.fk), float(self.gknorm), acc))
        else:
            self.recording((self.k, self.orcs, self.toc, 
                               float(self.fk), float(self.gknorm), acc))
                
    def oracleCalls(self):
        self.orcs += 2 * self.mini 
    
class Adam(Optimizer):
    
    def __init__(self, fun, x0, gradtol, maxite, maxorcs, mini, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8):
        self.info = SGD_STATS
        self.mini = mini
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = torch.zeros_like(x0, dtype = cTYPE, device = cCUDA)
        self.v = torch.zeros_like(x0, dtype = cTYPE, device = cCUDA)
        super().__init__(fun, x0, alpha, gradtol, maxite, maxorcs)
        
    def step(self):
        self.gk = self.fun(self.xk, "1")
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.gk
        self.v = self.beta2 * self.v + (1 - self.beta2) * (self.gk ** 2)
        mp = self.m / (1 - self.beta1 ** (self.k + 1))
        vp = self.v / (1 - self.beta2 ** (self.k + 1))
        self.xk -= self.alpha0 * mp / (torch.sqrt(vp) - self.epsilon)
        
    def recordStats(self, acc):
        if self.k == 0:
            self.gk = self.fun(self.xk, "1")
            self.fk = self.fun(self.xk, "f")
            #self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, 0, float(self.fk), float(self.gknorm), acc))
        else:
            if not self.k % 100:
                self.gknorm = torch.linalg.norm(self.gk, 2)
                self.recording((self.k, self.orcs, self.toc, 
                                float(self.fun(self.xk, "f")), float(self.gknorm), acc))
            
    def oracleCalls(self):
        self.orcs += 2 * self.mini 
        
class NewtonCG_TR_Steihaug(Optimizer):
    
    def __init__(self, fun, x0, gradtol, maxite, maxorcs, restol, inmaxite, 
                 deltaMax, delta0, eta, eta1, eta2, gamma1, gamma2, Hsub):
        
        if not (0 < eta1 and eta1 <= eta2 and eta2 < 1 and eta < eta1):
            raise Exception("etas 0 < eta < eta1 <= eta2 < 1")
        
        if not ((0 < gamma1 and gamma1 < 1) and (gamma2 > 1)):
            raise Exception("0 < gamma1 < 1 and gamma2 > 1")
        
        self.info = NEWTON_TR_STATS
        self.restol = restol
        self.inmaxite = inmaxite
        self.delta = delta0
        self.deltaMax = deltaMax
        self.eta = eta
        self.eta1 = eta1
        self.eta2 = eta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.Hsub = Hsub
        super().__init__(fun, x0, 0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk, self.dtype, m, self.inite = CGSteihaug(self.hk, self.gk, self.delta, self.restol, self.inmaxite)
        self.rho = (self.fk - self.fun(self.xk + pk, "0")) / m
        
        if self.rho < self.eta1:
            self.delta *= self.gamma1
        
        else:
            if self.rho > self.eta2 and self.dtype == "SOL,=":
                self.delta = min(self.delta * self.gamma2, self.deltaMax)
        
        if self.rho > self.eta:
            self.xk = self.xk + pk
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, "None", 0, 0, float(self.fk), 
                             float(self.gknorm), self.delta, acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((self.k, self.inite, self.dtype, self.orcs, 
                               self.toc, float(self.fk), float(self.gknorm), self.delta, acc))  
            
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite * self.Hsub + 2

