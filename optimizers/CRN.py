import torch 
from .optimizer import Optimizer
from .linesearchers.armijo import backwardArmijo

STATS = {"ite":"g", "orcs":"g", "time":".2f", "findM_Ite":"g", "cubicSolve":"g", "cubicOpt":".2e", "f":".4e", "g_norm":".4e", "acc":".2f"}
EPS = 1e-3
                   
class CubicRegNewton(Optimizer):
    
    def __init__(self, fun, x0, gradtol, maxite, maxorcs, alpha0, M0):
        self.info = STATS
        self.M = M0
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        self.xk, self.findMIte, self.cubicSolve, self.cubicOrc, self.cubicOpt = self.forwardtrackingCubic()
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        self.M /= 2
    
    def forwardtrackingCubic(self, FTCMax = 1000):
        xkp1, cfkp1, ite, total_oracle, cubicOpt = self.GDSolvesCubic(self.M, eps = EPS)
        fkp1 = self.fun(xkp1, "0")
        total_oracle += 1
        for i in range(FTCMax):
            if fkp1 <= cfkp1:
                return xkp1, i, ite, total_oracle, cubicOpt
            self.M *= 2
            xkp1, cfkp1, ite, oracles, cubicOpt = self.GDSolvesCubic(self.M, eps = EPS)
            fkp1 = self.fun(xkp1, "0")
            total_oracle += oracles + 1
    
    def GDSolvesCubic(self, M, eps = 1e-3, TMax = 10000):
        """
        Solve the cubic regularized subproblem using gradient descent.
        
        Parameters:
        M (float): The cubic regularization parameter.
        eps (float): The tolerance for the gradient norm.
        TMax (int): The maximum number of iterations.
        
        Returns:
        A tuple (yk, cfk, i, total_oracle, cubicOpt) where:
        - yk: The solution to the subproblem.
        - cfk: The function value at yk.
        - i: The number of iterations.
        - total_oracle: The total number of oracle calls.
        - cubicOpt: The optimal value of the cubic subproblem.
        """        
        # initialization 
        gknorm = torch.norm(self.gk)
        gknorm2 = gknorm ** 2
        gHg = torch.dot(self.gk, Av(self.hk, self.gk))
        gamma = - gHg / (M * gknorm2) + torch.sqrt((gHg / (M * gknorm2)) ** 2 + 2 * torch.sqrt(gknorm2) / (M))
        yk = self.xk - self.gk / torch.norm(self.gk) * gamma
        
        cfk, cgk = self.cubic_f(yk, M, order = "01") # 2 oracle calls
        total_oracle = 1
        eta = self.alpha0
        if torch.norm(cgk, torch.inf) < eps:
            return yk, cfk, 1, 2 * total_oracle, torch.norm(cgk, torch.inf)
        for i in range(TMax):
            # 2 * ite number of oracle calls
            eta, ite = backwardArmijo(lambda x : self.cubic_f(x, M, order = "0"), yk, cfk, cgk, eta, -cgk, 1e-4, 0.5, 100)
            total_oracle += ite
            yk = yk - eta * cgk
            eta *= 2
            cfk, cgk = self.cubic_f(yk, M, order = "01")
            if torch.norm(cgk) / gknorm  < eps:
                return yk, cfk, i + 2, 2 * total_oracle, torch.norm(cgk) / gknorm
        return yk, cfk, i + 2, 2 * total_oracle, torch.norm(cgk) / gknorm
            
    def cubic_f(self, y, M, order = "01"):
        """
        Compute the cubic regularized function value and gradient at point y.
        
        Parameters:
        y (torch.Tensor): The point at which to evaluate the function.
        M (float): The cubic regularization parameter.
        order (str): The order of derivatives to compute ("0" for function value, "01" for function value and gradient).
        
        Returns:
        If order == "0": returns the function value at y.
        If order == "01": returns a tuple (function value, gradient) at y.
        """
        ymx = y - self.xk
        norm_ymx = torch.norm(ymx)
        hk_ymx = Av(self.hk, ymx)
        cf = self.fk + torch.dot(self.gk, ymx) + torch.dot(hk_ymx, ymx) / 2 + M * (norm_ymx ** 3) / 6
        if "0" == order: 
            return cf
        if "01" == order:
            cg = self.gk + hk_ymx + M * norm_ymx * ymx / 2
            return cf, cg
            
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((0, 0, 0, 0, 0, 0, float(self.fk), float(self.gknorm), acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((self.k, self.orcs, self.toc, self.findMIte, self.cubicSolve, float(self.cubicOpt), float(self.fk), float(self.gknorm), acc))
    
    def oracleCalls(self):
        self.orcs += 2 + self.cubicOrc
        


def Av(A, v):
    if callable(A):
        return A(v)
    return torch.mv(A, v)
