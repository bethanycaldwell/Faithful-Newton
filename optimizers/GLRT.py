from galahad import glrt
import numpy as np
import torch 
from .optimizer import Optimizer

GLRT_STATS = {"ite":"g", "orcs":"g", "time":".2f", "findM_Ite":"g", "cubicSolve":"g", "cubicOpt":".2e", "f":".4e", "g_norm":".4e", "acc":".2f"}

class GLRT(Optimizer):
    
    def __init__(self, fun, x0, gradtol, maxite, maxorcs, alpha0):
        self.info = GLRT_STATS
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        # set default options
        options = glrt.initialize()
        # set problem data
        n = len(self.xk)
        power = 3.0
        weight = 1.0
        r = self.fun(self.xk, "0")
        v = np.empty(n)
        status = 1
        
        # solve problem using GLRT
        while status > 0:
            status, self.xk, r, v = glrt.solve_problem(status, n, power, weight, r, v)
            if status == 2: # precondition v -> v/2
                for i in range(n):
                    v[i] = v[i] / 2.0;
            elif status == 3: # form v -> H v
                hv[0] = 2.0 * v[0] + v[1]
                for i in range(1,n-1):
                    hv[i] = v[i-1] + 2.0 * v[i] + v[i+1]
                    hv[n-1] = v[n-2] + 2.0 * v[n-1]
                for i in range(n):
                    v[i] = hv[i]
            elif status == 4: # reset r
                r = self.fun(self.xk, "0")
        inform = glrt.information()

        # update xk and compute new function value and gradient
        #self.xk = torch.tensor(x)
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        glrt.terminate()

    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((0, 0, 0, 0, 0, float(self.fk), 
                             float(self.gknorm), 0, 0))
        else:
            self.gknorm = torch.linalg.norm(self.gk, torch.inf)
            self.recording((self.k, self.orcs, self.toc, 0, 0,
                            float(self.fk), float(self.gknorm), 0,
                            float(acc)))
    def oracleCalls(self):
        self.orcs += 1

    def ___main___():

        # set default options
        options = glrt.initialize()
        # set problem data
        n = len(self.xk)
        power = 3.0
        weight = 1.0
        r = self.fun(self.xk, "0")
        v = np.empty(n)
        status = 1
        
        # solve problem using GLRT
        while status > 0:
            status, self.xk, r, v = glrt.solve_problem(status, n, power, weight, r, v)
            if status == 2: # precondition v -> v/2
                for i in range(n):
                    v[i] = v[i] / 2.0;
            elif status == 3: # form v -> H v
                hv[0] = 2.0 * v[0] + v[1]
                for i in range(1,n-1):
                    hv[i] = v[i-1] + 2.0 * v[i] + v[i+1]
                    hv[n-1] = v[n-2] + 2.0 * v[n-1]
                for i in range(n):
                    v[i] = hv[i]
            elif status == 4: # reset r
                r = self.fun(self.xk, "0")
        inform = glrt.information()

        # update xk and compute new function value and gradient
        #self.xk = torch.tensor(x)
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        glrt.terminate()
