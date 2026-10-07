import torch, math
from hyperparameters import cCUDA, cTYPE
from .solvers.MSCR import OaMSNfo
from .optimizer import Optimizer
from .linesearchers.armijo import backwardArmijo

STATS = {"ite":"g", "orcs":"g", "time":".2f", "CRSolveIt":"g", "f":".4e", "g_norm":".4e", "acc":".2f"}
                   
class OptMS_CR(Optimizer):

    def __init__(self, fun, x0, gradtol, maxite, maxorcs, CRmaxit, maxbackIt, alpha0, lamb, sig, lazy):
        self.info = STATS
        # 1 < sig < 0
        self.sig, self.lambp, self.lazy = sig, lamb, lazy
        self.CRmaxit, self.maxIt = CRmaxit, maxbackIt
        self.vk, self.xk = x0, x0
        self.A, self.alpha = 0, alpha0
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
        
    def step(self):
        if self.k == 0:
            fk, gk, hk = self.fun(self.xk, "012")
            self.xk, self.lambp, self.OaMSNfoIte = OaMSNfo(hk, -gk, self.xk, self.fun, self.lambp, self.sig, self.lazy, self.CRmaxit, self.maxIt)
            self.akp1 = (1 + math.sqrt(1 + 4 * self.lambp * self.A)) / (2 * self.lambp)
            self.A = self.A + self.akp1
            self.lambp = self.lambp / self.alpha
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.vk = self.vk - self.akp1 * self.gk
        
        else:
            apkp1 = (1 + math.sqrt(1 + 4 * self.lambp * self.A)) / (2 * self.lambp)
            Apkp1 = self.A + apkp1
            yk = self.A * self.xk / Apkp1 + apkp1 * self.vk / Apkp1
            fk, gk, hk = self.fun(yk, "012")
            xpkp1, lamb, self.OaMSNfoIte = OaMSNfo(hk, -gk, yk, self.fun, self.lambp, self.sig, self.lazy, self.CRmaxit, self.maxIt)
            if self.lambp >= lamb:
                self.akp1, self.A = apkp1, Apkp1
                self.xk = xpkp1
                #print(">=",torch.norm(self.xk))
                self.lambp = self.lambp / self.alpha
            else:
                gamma = self.lambp / lamb
                self.akp1 = gamma * apkp1
                Akp1 = self.A + self.akp1
                self.xk = (1 - gamma) * self.A * self.xk / Akp1 + gamma * Apkp1 * xpkp1 / Akp1
                #print("<",torch.norm(xpkp1))
                self.lambp = self.alpha * self.lambp
                self.A = Akp1
                
            fk, gk = self.fun(xpkp1, "01")
            self.vk = self.vk - self.akp1 * gk
            self.fk, self.gk = self.fun(self.xk, "01")
            
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk = self.fun(self.xk, "01")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((0, 0, 0, 0, float(self.fk), float(self.gknorm), acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((self.k, self.orcs, self.toc, self.OaMSNfoIte, float(self.fk), float(self.gknorm), acc))
    
    def oracleCalls(self):
        self.orcs += 6 + 2 * self.OaMSNfoIte
        
def Av(A, v):
    if callable(A):
        return A(v)
    return torch.mv(A, v)