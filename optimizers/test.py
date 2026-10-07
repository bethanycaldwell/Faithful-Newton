# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:07:44 2024

@author: uqalim8
"""

import torch
from test_files.loss_funcs import logisticFun, logisticModel
from test_files.loadData import loadData
from .L_BFGS import L_BFGS
#from .NewtonMR_NC import NewtonMR_NC
#from .NewtonMR import NewtonMR
from .NewtonCG import NewtonCG
from .NewtonCR import NewtonCR
from .AdaN import AdaN
from .CRN import CubicRegNewton

if "__main__" == __name__:
    A_train, b_train, *_ = loadData()
    fun = lambda x, v : logisticFun(x, A_train, b_train, 1, v)
    x0 = torch.zeros(A_train.shape[-1], dtype = torch.float64)
    Lg = torch.linalg.matrix_norm(A_train, 2)**2 / 4 + 1
    
    def pred(x):
        b_pred = torch.round(logisticModel(A_train, x))
        return torch.sum(b_train == b_pred) / len(b_train) * 100
    
    # print("=================== LineSearch GD ========================")
    # optGD = linesearchGD(fun, x0.clone(), 10/Lg, 10e-4, 1000, 1000, 100, 10e-4, 0.9)
    # optGD.optimize(True, pred)
    
    # print("=================== Adam ========================")
    # optAD = Adam(fun, x0.clone(), 1e-4, 1000, 1000)
    # optAD.optimize(True, pred)
    
    # print("===================== AdaN ==========================")
    # optAdaN = AdaN(fun, x0.clone(), 1e-4, 1000, 10000, H0 = 0, AdaN = "+")
    # optAdaN.optimize(True, pred)
    
    print("=================== Cubic ========================")
    optCubic = CubicRegNewton(fun, x0.clone(), 1e-4, 100, 10000, M0 = 1e-4)
    optCubic.optimize(True, pred)
    
    print("=================== Newton CG ========================")
    optNEWTONCG = NewtonCG(fun, x0.clone(), 1, 1e-4, 100, 10000, 1e-3, 100, 100, 10e-4, 0.9)
    optNEWTONCG.optimize(True, pred)

    # print("=================== Newton MR ========================")
    # optNEWTONMR = NewtonMR(fun, x0.clone(), 1, 10e-4, 100, 10000, 10e-6, 100, 100, 10e-4, 0.9)
    # optNEWTONMR.optimize(True, pred)
    
    print("=================== Newton CR ========================")
    optNEWTONCR = NewtonCR(fun, x0.clone(), 1, 1e-4, 100, 10000, 1e-9, 100, 100, 10e-4, 0.9)
    optNEWTONCR.optimize(True, pred)
    
    # print("=================== Newton MR NC ========================")
    # optNEWTONMR_NC = NewtonMR_NC(fun, x0.clone(), 1, 1e-4, 100, 10000, 10e-6, 100, 100, 10e-4, 0.9, 10e-4, 1)
    # optNEWTONMR_NC.optimize(True, pred)
    
    # print("=================== Newton CG NC ========================")
    # optNEWTONCG_NC = NewtonCG_NC(fun, x0.clone(), 1, 1e-4, 1000, 1000, 10e-2, 1000, 1000, 10e-4, 0.9, 10e-4, 1)
    # optNEWTONCG_NC.optimize(True, pred)
    
    print("=================== L-BFGS ========================")
    optL_BFGS = L_BFGS(fun, x0.clone(), 1, 1e-4, 20, 1000, 10000, 1000)
    optL_BFGS.optimize(True, pred)
    
    # print("=================== Newton TR ========================")
    # optNewtonTR = NewtonCG_TR_Steihaug(fun, x0.clone(), 10e-4, 1000, 1000, 
    #                                    10e-2, 1000, 1e10, 1e5, 1/8, 0.25, 0.75, 0.25, 2, 1)
    # optNewtonTR.optimize(True, pred)
    
    # import matplotlib.pyplot as plt
    
    # fig = plt.figure()
    # plt.loglog(torch.tensor(optGD.record["orcs"]) + 1, optGD.record["f"], label = "GD")
    # plt.loglog(torch.tensor(optAD.record["orcs"]) + 1, optAD.record["f"], label = "Adam")
    # plt.loglog(torch.tensor(optNEWTONCG.record["orcs"]) + 1, optNEWTONCG.record["f"], label = "NewtonCG")
    # plt.loglog(torch.tensor(optNEWTONMR.record["orcs"]) + 1, optNEWTONMR.record["f"], label = "NewtonMR")
    # plt.loglog(torch.tensor(optNEWTONCR.record["orcs"]) + 1, optNEWTONCR.record["f"], label = "NewtonCR")
    # plt.loglog(torch.tensor(optNEWTONCG_NC.record["orcs"]) + 1, optNEWTONCG_NC.record["f"], label = "NewtonCG_NC")
    # plt.loglog(torch.tensor(optNewtonTR.record["orcs"]) + 1, optNewtonTR.record["f"], label = "NewtonTR")
    # plt.loglog(torch.tensor(optL_BFGS.record["orcs"]) + 1, optL_BFGS.record["f"], label = "L_BFGS")

    # plt.legend()
    # plt.show()
    
    # fig = plt.figure()
    # plt.loglog(torch.tensor(optGD.record["orcs"]) + 1, optGD.record["g_norm"], label = "GD")
    # plt.loglog(torch.tensor(optAD.record["orcs"]) + 1, optAD.record["g_norm"], label = "Adam")
    # plt.loglog(torch.tensor(optNEWTONCG.record["orcs"]) + 1, optNEWTONCG.record["g_norm"], label = "NewtonCG")
    # plt.loglog(torch.tensor(optNEWTONMR.record["orcs"]) + 1, optNEWTONMR.record["g_norm"], label = "NewtonMR")
    # plt.loglog(torch.tensor(optNEWTONCR.record["orcs"]) + 1, optNEWTONCR.record["g_norm"], label = "NewtonCR")
    # plt.loglog(torch.tensor(optNEWTONCG_NC.record["orcs"]) + 1, optNEWTONCG_NC.record["g_norm"], label = "NewtonCG_NC")
    # plt.loglog(torch.tensor(optNewtonTR.record["orcs"]) + 1, optNewtonTR.record["g_norm"], label = "NewtonTR")
    # plt.loglog(torch.tensor(optL_BFGS.record["orcs"]) + 1, optL_BFGS.record["g_norm"], label = "L_BFGS")

    # plt.legend()
    # plt.show()