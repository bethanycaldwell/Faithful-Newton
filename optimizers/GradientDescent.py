import torch 
from .optimizer import Optimizer
from .linesearchers.armijo import backwardArmijo
from hyperparameters import cTYPE, cCUDA

GD_STATS = {"ite":"g", "orcs":"g", "time":".2f", "f":".4e", 
            "g_norm":".4e", "alpha":".2e", "acc":".2f"}

class linesearchGD(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, lineMaxite, lineBetaB, lineRho):
        self.info = GD_STATS
        self.lineMaxite = lineMaxite
        self.lineBetaB = lineBetaB
        self.lineRho = lineRho
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk = -self.gk
        self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), self.xk, self.fk, self.gk, self.alphak, pk,
                                     self.lineBetaB, self.lineRho, self.lineMaxite)
      
        self.xk += self.alphak * pk
        self.alphak *= 2
        self.fk, self.gk = self.fun(self.xk, "01")
        self.gknorm = torch.linalg.norm(self.gk, torch.inf)

    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk = self.fun(self.xk, "01")
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((0, 0, 0, float(self.fk), float(self.gknorm), 0, acc))
        else:
            self.recording((self.k, self.orcs, self.toc, 
                               float(self.fk), float(self.gknorm), self.alphak, acc))
        
    def oracleCalls(self):
        self.orcs += 2 + self.lineite