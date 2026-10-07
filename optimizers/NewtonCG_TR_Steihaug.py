import torch 
from .optimizer import Optimizer
from .linesearchers.armijo import backwardArmijo
from .solvers.CGSteihaug import CGSteihaug

NEWTON_TR_STATS = {"ite":"g", "inite":"g", "dtype":"", "orcs":"g", "time":".2f",
                   "f":".4e", "g_norm":".4e", "delta":".2e", "acc":".2f"}
                   
class NewtonCG_TR_Steihaug(Optimizer):
    
    def __init__(self, fun, x0, gradtol, maxite, maxorcs, restol, inmaxite, 
                 deltaMax, delta0, eta, eta1, eta2, gamma1, gamma2):
        
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
            self.recording((0, 0, "None", 0, 0, float(self.fk), float(self.gknorm), self.delta, acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((self.k, self.inite, self.dtype, self.orcs, self.toc, float(self.fk), 
                            float(self.gknorm), self.delta, acc))  
            
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite + 2