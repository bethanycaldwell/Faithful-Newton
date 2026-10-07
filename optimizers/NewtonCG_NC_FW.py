# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:49:37 2024

@author: uqalim8
"""

import torch
from NewtonCG_NC import NewtonCG_NC
from CG import CappedCG
from linesearchers.cubic import dampedNewtonCGLinesearch, dampedNewtonCGbackForwardLS

class NewtonCG_NC_FW(NewtonCG_NC):

    def step(self):
        pk, self.dtype, self.inite, pHp, _ = CappedCG(self.hk, -self.gk, self.restol, self.epsilon, self.inmaxite)
        normpkcubed = torch.linalg.norm(pk, 2)**3
        if self.dtype == "NC":
            pk = - torch.sign(torch.dot(pk, self.gk)) * abs(pHp) * pk / normpkcubed
            self.alphak, self.lineite = dampedNewtonCGbackForwardLS(lambda x : self.fun(x, "0"), self.xk, self.fk, self.alpha0, pk, 
                                                                 normpkcubed, self.lineBeta, self.lineRho, self.lineMaxite)
        else:
            self.alphak, self.lineite = dampedNewtonCGLinesearch(lambda x : self.fun(x, "0"), self.xk, self.fk, self.alpha0, pk, 
                                                             normpkcubed, self.lineBeta, self.lineRho, self.lineMaxite)
        
        self.xk += self.alphak * pk
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")