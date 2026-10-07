# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:01:15 2024

@author: uqalim8
"""

from .NewtonCG_TR_Steihaug import NewtonCG_TR_Steihaug
from .faithfulNewtonCG import FaithfulNewtonCG
from .faithfulNewtonCR import FaithfulNewtonCR
from .faithfulNewtonCR_reg import FaithfulNewtonCR_reg
from .GradientDescent import linesearchGD
from .NewtonMR_NC import NewtonMR_NC
from .NewtonCR_NC import NewtonCR_NC
from .NewtonCG_NC import NewtonCG_NC
from .NewtonCG import NewtonCG
from .NewtonCR import NewtonCR
from .L_BFGS import L_BFGS
from .AdaN import AdaN
from .AccCRN import AccCubicRegNewton
from .CRN import CubicRegNewton
from .NATA import AccCRNAdapt
from .OptMS import OptMS_CR


TEXT = "{:<20} : {:>20}"

def init_algorithms(fun, x0, algo, c):
    
    if algo == "FaithfulNewtonCR":
        print(TEXT.format("Algorithm", algo))
        return FaithfulNewtonCR(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.beta, c.skips, c.T,
                                c.inmaxite, c.lineMaxite, c.lineBetaB, c.lineRho, c.reOrtho)
    
    if algo == "FaithfulNewtonCR-reg":
        print(TEXT.format("Algorithm", algo))
        return FaithfulNewtonCR_reg(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.beta, c.skips, c.T, c.reg,
                                    c.inmaxite, c.lineMaxite, c.lineBetaB, c.lineRho, c.reOrtho)
    
    if algo == "FaithfulNewtonCG":
        print(TEXT.format("Algorithm", algo))
        return FaithfulNewtonCG(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.beta, c.skips, c.T,
                                c.inmaxite, c.lineMaxite, c.lineBetaB, c.lineRho)
    
    if algo == "NewtonCG":
        print(TEXT.format("Algorithm", algo))
        return NewtonCG(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.restol, c.inmaxite, 
                        c.lineMaxite, c.lineBeta, c.lineRho, c.reOrtho)
    
    if algo == "NewtonCR":
        print(TEXT.format("Algorithm", algo))
        return NewtonCR(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.restol, c.inmaxite, 
                        c.lineMaxite, c.lineBeta, c.lineRho)
        
    if algo == "NewtonMR-NC":
        print(TEXT.format("Algorithm", algo))
        return NewtonMR_NC(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.restol, c.inmaxite, 
                           c.lineMaxite, c.lineBetaB, c.lineRho, c.lineBetaFB)
    
    if algo == "NewtonCR-NC":
        print(TEXT.format("Algorithm", algo))
        return NewtonCR_NC(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.restol, c.inmaxite, 
                           c.lineMaxite, c.lineBetaB, c.lineRho, c.lineBetaFB)                   
    
    if algo == "NewtonCappedCG":
        print(TEXT.format("Algorithm", algo))
        return NewtonCG_NC(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.restol, c.inmaxite, 
                           c.lineMaxite, c.lineBeta, c.lineRho, c.epsilon)
                           
    if algo == "L-BFGS":
        print(TEXT.format("Algorithm", algo))
        return L_BFGS(fun, x0, c.alpha0, c.gradtol, c.m, c.maxite, c.maxorcs, c.lineMaxite)
    
    if algo == "TR_Steihaug":
        print(TEXT.format("Algorithm", algo))
        return NewtonCG_TR_Steihaug(fun, x0, c.gradtol, c.maxite, c.maxorcs, c.restol, c.inmaxite, 
                                    c.deltaMax, c.delta0, c.eta, c.eta1, c.eta2, c.gamma1, c.gamma2)

    if algo == "AdaN+":
        print(TEXT.format("Algorithm", algo))
        return AdaN(fun, x0, c.gradtol, c.maxite, c.maxorcs, c.H0, "+")
        
    if algo == "AdaN":
        print(TEXT.format("Algorithm", algo))
        return AdaN(fun, x0, c.gradtol, c.maxite, c.maxorcs, c.H0, "-")
 
    if algo == "linesearchGD":
        print(TEXT.format("Algorithm", algo))
        return linesearchGD(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.lineMaxite, c.lineBetaB, c.lineRho)
        
    if algo == "CRN":
        print(TEXT.format("Algorithm", "CubicRegNewton"))
        return CubicRegNewton(fun, x0, c.gradtol, c.maxite, c.maxorcs, c.alpha0, c.M0)
        
    if algo == "AccCRN":
        print(TEXT.format("Algorithm", "AccCubicRegNewton"))
        return AccCubicRegNewton(fun, x0, c.gradtol, c.maxite, c.maxorcs, c.alpha0, c.M0)
        
    if algo == "NATA":
        print(TEXT.format("Algorithm", "NATA"))
        return AccCRNAdapt(fun, x0, c.gradtol, c.maxite, c.maxorcs, c.alpha0, c.M0, c.nu0, c.nuMin, c.nuMax, c.theta)
        
    if algo == "OptMS":
        print(TEXT.format("Algorithm", "OptMS_CR"))
        return OptMS_CR(fun, x0, c.gradtol, c.maxite, c.maxorcs, c.CRmaxit, c.maxbackIt, c.alpha0, c.lamb, c.sig, c.lazy)