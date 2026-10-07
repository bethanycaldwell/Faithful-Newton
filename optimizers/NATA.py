import torch 
from hyperparameters import cCUDA, cTYPE
from .optimizer import Optimizer
from .linesearchers.armijo import backwardArmijo

STATS = {"ite":"g", "orcs":"g", "time":".2f", "cubicSolve":"g", "cubicOpt":".2e", "f":".4e", "g_norm":".4e", "acc":".2f"}
EPS = 1e-3                   
class AccCRNAdapt(Optimizer):

    def __init__(self, fun, x0, gradtol, maxite, maxorcs, alpha0, M0, nu0, nuMin, nuMax, theta):
        self.info = STATS
        self.M = M0
        self.x0, self.xk, self.vk = x0, x0, x0
        self.sk = torch.zeros_like(x0, dtype = cTYPE, device = cCUDA)
        self.fpsi, self.gkxk = 0, 0
        self.A, self.nuMin, self.nuMax, self.nu, self.theta = 0, nuMin, nuMax, nu0, theta
        self.tas, self.fs, self.gs, self.gdotxs = None, None, None, None
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        self.cubicOracles = 0
        first_cycle = True
        xk = self.xk
        vk = self.vk
        for i in range(1000):
            new_nu = max(self.nu / self.theta, self.nuMin)
            
            # NATA diverges
            if new_nu == self.nuMin:
                self.orcs = 1e10
                return 
                
            self.nu = new_nu
            ta = self.nu * ((self.k + 1) ** 3 - self.k ** 3) / self.M
            ta = torch.tensor(ta, dtype = cTYPE, device = cCUDA)
            Akp1 = self.A + ta
            yk = self.A * self.xk / Akp1 + ta * self.vk / Akp1
            xk, self.cubicIte, self.cubicOpt, total_cubic_oracle = self.GDSolvesCubic(yk, self.M, eps = EPS)
            self.cubicOracles += total_cubic_oracle
            self.fk, self.gk = self.fun(xk, "01")
            
            sk = self.sk + ta * self.gk
            fpsi = self.fpsi + ta * self.fk
            gkxk = self.gkxk + ta * torch.dot(self.gk, xk) 
            sk_norm_sq = torch.sqrt(torch.norm(sk))
            vk = self.x0 - sk / sk_norm_sq
            
            if torch.dot(sk, vk) + fpsi - gkxk + (torch.norm(vk - self.x0) ** 3) / 3 > Akp1 * self.fk:
                self.A = Akp1
                self.cubicOracles += 2 * (i + 1)
                self.xk = xk
                self.vk = vk
                self.sk += ta * self.gk
                self.fpsi += ta * self.fk
                self.gkxk += ta * torch.dot(self.gk, self.xk)
                self.nu = min(self.nu * self.theta ** 2, self.nuMax)
                return
    
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
            self.recording((0, 0, 0, 0, 0, float(self.fk), float(self.gknorm), acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((self.k, self.orcs, self.toc, self.cubicIte, 
                            float(self.cubicOpt), float(self.fk), float(self.gknorm), acc))
    
    def oracleCalls(self):
        self.orcs += self.cubicOracles
        
def Av(A, v):
    if callable(A):
        return A(v)
    return torch.mv(A, v)