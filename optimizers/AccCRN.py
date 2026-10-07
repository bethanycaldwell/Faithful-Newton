import torch 
from hyperparameters import cCUDA, cTYPE
from .optimizer import Optimizer
from .linesearchers.armijo import backwardArmijo

STATS = {"ite":"g", "orcs":"g", "time":".2f", "cubicSolve":"g", "cubicOpt":".2e", "f":".4e", "g_norm":".4e", "acc":".2f"}

EPS = 1e-3

class AccCubicRegNewton(Optimizer):
    """
    Accelerating the cubic regularization of Newton's method on convex problems
    
    Yu. Nesterov
    
    Algorithm (4.8)
    """
    def __init__(self, fun, x0, gradtol, maxite, maxorcs, alpha0, M0):
        self.info = STATS
        self.M = M0
        self.vk, self.x0, self.xk = x0, x0, x0
        self.sk = torch.zeros_like(x0, dtype = cTYPE, device = cCUDA)
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
    
        # initialization
        if self.k == 0:
            self.xk, self.cubicIte, self.cubicOpt, self.total_cubic_oracle = self.GDSolvesCubic(self.x0, self.M, eps = EPS)
            self.fk, self.gk = self.fun(self.xk, "01")
            
        elif self.k == 1:
            self.yk = self.xk / 4 + 3 * self.vk / 4
            self.xk, self.cubicIte, self.cubicOpt, self.total_cubic_oracle = self.GDSolvesCubic(self.yk, 2 * self.M, eps = EPS)
            
            self.fk, self.gk = self.fun(self.xk, "01")
            self.sk += (self.k + 1) * (self.k + 2) * self.gk / 2
            #self.fs = torch.cat((self.fs, self.fk.reshape(1)))
            #self.gs = self.gk.reshape(-1,1)
            #self.gdotxs = torch.dot(self.xk, self.gk).reshape(1)      
            
        else:
            #self.vk, self.auxIte, self.auxOpt = self.GDSolvesAuxF(6 * self.M, eps = EPS)
            sk_norm = torch.norm(self.sk)
            self.vk = self.x0 - self.sk / torch.sqrt(3 * self.M * sk_norm) 
            self.yk = self.k * self.xk / (self.k + 3) + 3 * self.vk / (self.k + 3) 
            self.xk, self.cubicIte, self.cubicOpt, self.total_cubic_oracle = self.GDSolvesCubic(self.yk, 2 * self.M, eps = EPS)
            
            self.fk, self.gk = self.fun(self.xk, "01")
            self.sk += (self.k + 1) * (self.k + 2) * self.gk / 2
            #self.fs = torch.cat((self.fs, self.fk.reshape(1)))
            #self.gs = torch.cat((self.gs, self.gk.reshape(-1,1)), dim = 1)
            #self.gdotxs = torch.cat((self.gdotxs, torch.dot(self.xk, self.gk).reshape(1)))
        
#    def GDSolvesAuxF(self, M, eps = 1e-3, TMax = 10000):
#        # initialization 
#        vk = self.vk
#        
#        cfk, cgk = self.auxF(vk, self.x0, self.gs, self.fs, self.gdotxs, M, order = "01") # 2 oracle calls
#        eta = self.alpha0
#        if torch.norm(cgk, torch.inf) < eps:
#            return vk, 1, torch.norm(cgk, torch.inf)
#        for i in range(TMax):
#            # 2 * ite number of oracle calls
#            eta, ite = backwardArmijo(lambda x : self.auxF(x, self.x0, self.gs, self.fs, self.gdotxs, M, order = "0"), 
#                                      vk, cfk, cgk, eta, -cgk, 1e-4, 0.5, 100)
#            vk = vk - eta * cgk
#            eta *= 2
#            cfk, cgk = self.auxF(vk, self.x0, self.gs, self.fs, self.gdotxs, M, order = "01") 
#            if torch.norm(cgk, torch.inf) < eps:
#                return vk, i + 1, torch.norm(cgk, torch.inf)
#        return vk, i + 1, torch.norm(cgk, torch.inf)
#    
#    def auxF(self, x, x0, grads, fs, graddotxs, C, order = "01"):
#        _, k = grads.shape
#        normxmx0 = torch.norm(x - x0)
#        first_term = fs[0] + C * (normxmx0 ** 3) / 6
#        
#        gdotx = torch.einsum("ij,i->j", grads, x)
#        K = torch.arange(k, dtype = cTYPE, device = cCUDA) + 1
#        K = (K + 1) * (K + 2) / 2 
#        af = first_term + torch.sum(K * (fs[1:] + gdotx - graddotxs))
#        if order == "0":
#            return af
#        gf = torch.einsum("ij->i", K * grads).flatten() + C * normxmx0 * (x - x0) / 2
#        return af, gf
    
    def GDSolvesCubic(self, h0, M, eps = 1e-3, TMax = 10000):
        # initialization
        fyk, gyk, hyk = self.fun(h0, "012")
        gknorm = torch.norm(gyk)
        gknorm2 = gknorm ** 2
        gHg = torch.dot(gyk, Av(hyk, gyk))
        gamma = - gHg / (M * gknorm2) + torch.sqrt((gHg / (M * gknorm2)) ** 2 + 2 * torch.sqrt(gknorm2) / (M))
        yk = h0 - gyk / torch.norm(gyk) * gamma
        
        cfk, cgk = self.cubic_f(yk, h0, M, fyk, gyk, hyk, order = "01") # 2 oracle calls
        total_oracle = 1
        eta = self.alpha0
        if torch.norm(cgk, torch.inf) < eps:
            return yk, 1, torch.norm(cgk, torch.inf), 2 + 2 * total_oracle
        for i in range(TMax):
            # 2 * ite number of oracle calls
            eta, ite = backwardArmijo(lambda x : self.cubic_f(x, h0, M, fyk, gyk, hyk, order = "0"), 
                                      yk, cfk, cgk, eta, -cgk, 1e-4, 0.5, 100)
            total_oracle += ite
            yk = yk - eta * cgk
            eta *= 2
            cfk, cgk = self.cubic_f(yk, h0, M, fyk, gyk, hyk, order = "01")
            if torch.norm(cgk) / gknorm < eps:
                return yk, i + 2, torch.norm(cgk) / gknorm, 2 * total_oracle
        return yk, i + 2, torch.norm(cgk) / gknorm, 2 + 2 * total_oracle
            
    def cubic_f(self, y, h0, M, fyk, gyk, hyk, order = "01"):
        ymx = y - h0
        norm_ymx = torch.norm(ymx)
        hk_ymx = Av(hyk, ymx)
        cf = fyk + torch.dot(gyk, ymx) + torch.dot(hk_ymx, ymx) / 2 + M * (norm_ymx ** 3) / 6
        if "0" == order: 
            return cf
        if "01" == order:
            cg = gyk + hk_ymx + M * norm_ymx * ymx / 2
            return cf, cg
            
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk = self.fun(self.xk, "01")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((0, 0, 0, 0, float(0), float(self.fk), float(self.gknorm), acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((self.k, self.orcs, self.toc, self.cubicIte, 
                            float(self.cubicOpt), float(self.fk), float(self.gknorm), acc))
    
    def oracleCalls(self):
        self.orcs += 2 + self.total_cubic_oracle

def Av(A, v):
    if callable(A):
        return A(v)
    return torch.mv(A, v)