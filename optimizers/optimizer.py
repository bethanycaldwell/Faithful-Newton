# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:47:36 2024

@author: uqalim8
"""

import time

class Optimizer:
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs):
        self.fun = fun
        self.xk = x0
        self.alpha0 = alpha0
        self.maxorcs = maxorcs
        self.k, self.orcs, self.toc, self.lineite = 0, 0, 0, 0
        self.gknorm, self.record = None, None
        self.maxite = maxite
        self.gradtol = gradtol
        self.alphak = 1
        self.record = dict(((i, []) for i in self.info.keys()))
        
    def recording(self, stats):
        for n, i in enumerate(self.record.keys()):
            self.record[i].append(stats[n])
            
    def printStats(self):
        if self.k == 0:
            print(7 * len(self.info) * "..")
            form = ["{:^13}"] * len(self.info)
            print("|".join(form).format(*self.info.keys()))
            print(7 * len(self.info) * "..")
        form = ["{:^13" + i + "}" for i in self.info.values()]
        print("|".join(form).format(*(self.record[i][-1] for i in self.info.keys())))        
    
    def progress(self, verbose, pred, print_skip = 1):
        self.k += 1
        self.oracleCalls()
        self.recordStats(pred(self.xk))
        if verbose and self.k % verbose == 0:
            self.printStats()

    def termination(self):
        return self.k > self.maxite or self.gknorm < self.gradtol or self.orcs > self.maxorcs or self.alphak < 1e-18
    
    def optimize(self, verbose, pred = lambda x : 0):
        self.recordStats(pred(self.xk))
        self.printStats()
        while not self.termination():
            tic = time.time()
            self.step()
            self.toc += time.time() - tic
            self.progress(verbose, pred)

    def recordStats(self):
        raise NotImplementedError
        
    def step(self):
        raise NotImplementedError
    
    def oracleCalls(self):
        raise NotImplementedError
    