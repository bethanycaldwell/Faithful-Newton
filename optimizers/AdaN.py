import torch 
from .optimizer import Optimizer
from .solvers.AdaNCG import AdaNCG
from hyperparameters import cTYPE, cCUDA

NEWTON_STATS = {"ite":"g", "orcs":"g", "time":".2f", "relr":".4e", "lite":"g",
                "f":".4e", "g_norm":".4e", "acc":".2f"}
                   
class AdaN(Optimizer):
    
    def __init__(self, fun, x0, gradtol, maxite, maxorcs, H0, AdaN = "+"):
        self.info = NEWTON_STATS
        self.restol = 1e-32
        self.H = H0
        self.AdaN = AdaN
        self.N = 100
        self.i = 0
        super().__init__(fun, x0, 0, gradtol, maxite, maxorcs)
        if self.AdaN == "+":
            self.xm1 = x0.clone() + torch.rand(x0.shape[0], dtype = cTYPE, device = cCUDA) * 0.01
            _, g0, h0 = fun(x0, "012")
            _, self.gm1 = fun(self.xm1, "01")
            self.H = torch.norm(g0 - self.gm1 - Av(h0, x0 - self.xm1)) / (torch.norm(x0 - self.xm1) ** 2)
            self.M = self.H
            self.orcs += 2
        
    def step(self):
        if self.AdaN == "+":
            if self.k != 0:
                self.M = torch.norm(self.gk - self.gm1 - Av(self.hk, self.xk - self.xm1)) / (torch.norm(self.xk - self.xm1) ** 2)
            self.H = torch.max(self.M, self.H / 2)
            self.reg = torch.sqrt(self.H * self.gknorm)
            pk, self.inite, self.relr = AdaNCG(lambda v : Av(self.hk, v) + self.reg * v, -self.gk, 
                                           self.restol, self.gk.shape[0] + 1, reOrtho = False)
            self.xm1 = self.xk.clone()
            self.gm1 = self.gk.clone()
            self.xk = self.xk + pk
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        
        else:
            self.inite = 0
            self.H /= 4
            for self.i in range(self.N):
                self.H *= 2 
                self.reg = torch.sqrt(self.H * self.gknorm)
                pk, k, self.relr = AdaNCG(lambda v : self.hk(v) + self.reg * v, -self.gk, 
                                  self.restol, self.gk.shape[0] + 1, reOrtho = False)
                xk_test = self.xk + pk
                r = torch.norm(pk)
                f_test, g_test, hk_test = self.fun(xk_test, "012")
                self.inite += (k + 1)
                if f_test <= self.fk - 2 * self.reg * (r ** 2) / 3 and torch.norm(g_test) <= 2 * self.reg * r:
                    self.xk = xk_test
                    self.fk, self.gk, self.hk = f_test, g_test, hk_test
                    break
                elif self.i == self.N - 1:
                    # force termination
                    self.alphak = -1
                
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((0, 0, 0, 0, 0, float(self.fk), float(self.gknorm), acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((self.k, self.orcs, self.toc, float(self.relr), self.i + 1, float(self.fk), float(self.gknorm), acc))
            
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite
        
def Av(A, v):
    if callable(A):
        return A(v)
    return torch.mv(A, v)
