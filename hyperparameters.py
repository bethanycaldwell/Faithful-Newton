# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:23:10 2023

@author: uqalim8
"""
import torch

cTYPE = torch.float64
cCUDA = True
GRADTOL = 1e-6
LINEBETA = 1e-4
INMAX = 10000
MAXITE = 5000
MAXORC = 5e4
LINEMAX = 100
BACKTRACK = 0.5
NEWTONALPHA0 = 1
REORTHO = False

if cCUDA:
    cCUDA = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    cCUDA = torch.device("cpu")

class const():
    pass

# Faithful-Newton
# Example: FaithfulNewtonCG, FaithfulNewtonCR
cFN = const()
cFN.alpha0 = NEWTONALPHA0
cFN.gradtol = GRADTOL
cFN.maxite = MAXITE
cFN.inmaxite = INMAX
cFN.beta = 0.01
cFN.skips = 20
cFN.T = 5
cFN.reg = 0.01 # only relevant to faithfulNewton with regualarized Hessian
cFN.maxorcs = MAXORC
cFN.lineMaxite = LINEMAX
cFN.lineBetaB = LINEBETA
cFN.lineRho = BACKTRACK
cFN.reOrtho = REORTHO

# Newton's method with line-search
# Examples: NewtonCG, NewtonCR and NewtonMR
cNWL = const()
cNWL.alpha0 = NEWTONALPHA0
cNWL.gradtol = GRADTOL
cNWL.maxite = MAXITE
cNWL.maxorcs = MAXORC
cNWL.inmaxite = INMAX
cNWL.restol = 0.1
cNWL.lineMaxite = LINEMAX
cNWL.lineBeta = LINEBETA
cNWL.lineRho = BACKTRACK
cNWL.reOrtho = REORTHO

# Stephen Wright's NewtonCappedCG
cCCG = const()
cCCG.alpha0 = NEWTONALPHA0
cCCG.gradtol = GRADTOL
cCCG.maxite = MAXITE
cCCG.maxorcs = MAXORC
cCCG.restol = 1e-3
cCCG.inmaxite = INMAX
cCCG.lineMaxite = LINEMAX
cCCG.lineBeta = BACKTRACK
cCCG.lineRho = BACKTRACK
cCCG.epsilon = 0.99

# Newton's method with NPC detection and forward / backward linesearch
cMCRNPC = const()
cMCRNPC.alpha0 = NEWTONALPHA0
cMCRNPC.gradtol = GRADTOL
cMCRNPC.maxite = MAXITE
cMCRNPC.restol = 1
cMCRNPC.inmaxite = INMAX
cMCRNPC.maxorcs = MAXORC
cMCRNPC.lineMaxite = LINEMAX
cMCRNPC.lineBetaB = LINEBETA
cMCRNPC.lineRho = BACKTRACK
cMCRNPC.lineBetaFB = LINEBETA

# GradientRegularizedNewton e.g., AdaN, AdaN+
cAdaN = const()
cAdaN.gradtol = GRADTOL
cAdaN.maxite = MAXITE
cAdaN.maxorcs = MAXORC
cAdaN.H0 = 0.01

# CubicRegNewton / AccCubicRegNewton
cCubic = const()
cCubic.gradtol = GRADTOL
cCubic.maxite = MAXITE
cCubic.maxorcs = MAXORC
cCubic.alpha0 = 0.5
cCubic.M0 = 1.5e3

cNATA = const()
cNATA.gradtol = GRADTOL
cNATA.maxite = MAXITE
cNATA.maxorcs = MAXORC
cNATA.alpha0 = 0.5
cNATA.theta = 2
cNATA.nuMin = 1e-8
cNATA.nuMax = 100
cNATA.nu0 = cNATA.nuMax * cNATA.theta
cNATA.M0 = 1.5e3

cMS = const()
cMS.gradtol = GRADTOL
cMS.maxite = MAXITE
cMS.maxorcs = MAXORC
cMS.CRmaxit = INMAX
cMS.maxbackIt = INMAX
cMS.alpha0 = 2
cMS.lamb = 1.1
cMS.sig = 0.5
cMS.lazy = True

cGD = const()
cGD.alpha0 = 0.1
cGD.gradtol = GRADTOL
cGD.maxite = MAXITE
cGD.maxorcs = MAXORC
cGD.lineMaxite = LINEMAX
cGD.lineBetaB = LINEBETA
cGD.lineRho = BACKTRACK

cTR_STEI = const()
cTR_STEI.gradtol = GRADTOL
cTR_STEI.maxite = MAXITE
cTR_STEI.inmaxite = INMAX
cTR_STEI.maxorcs = MAXORC
cTR_STEI.restol = 0.1                          
cTR_STEI.deltaMax = 1e10
cTR_STEI.delta0 = 10
cTR_STEI.eta = 0.05
cTR_STEI.eta1 = 1/4
cTR_STEI.eta2 = 3/4
cTR_STEI.gamma1 = 1/4
cTR_STEI.gamma2 = 2

cL_BFGS = const()
cL_BFGS.alpha0 = NEWTONALPHA0
cL_BFGS.gradtol = GRADTOL
cL_BFGS.m = 20
cL_BFGS.maxite = MAXITE
cL_BFGS.maxorcs = MAXORC
cL_BFGS.lineMaxite = LINEMAX

cADAM = const()
cADAM.alpha0 = 0.00001
cADAM.beta1 = 0.9 #0.9
cADAM.beta2 = 0.999
cADAM.epsilon = 1e-8
cADAM.gradtol = GRADTOL
cADAM.maxite = MAXITE
cADAM.maxorcs = MAXORC

cSGD = const()
cSGD.alpha0 = 0.5
cSGD.gradtol = GRADTOL
cSGD.maxite = MAXITE
cSGD.maxorcs = MAXORC
