# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:29:32 2024

@author: uqalim8
"""

import torch

def backwardArmijo(fun, xk, fk, gk, alpha, pk, beta, rho, maxite, minalpha = 1e-14):
    betagp = torch.dot(gk, pk) * beta
    assert betagp < 0
    j = 1
    while fun(xk + alpha * pk) > fk + alpha * betagp and j < maxite:
        alpha *= rho
        j += 1
        if alpha < minalpha:
            alpha = minalpha
            break
    if j >= maxite:
        print("linesearch max exceeded!")
    return alpha, j

def backForwardArmijo_Yang(fun, xk, fk, gk, alpha, pk, beta, rho, maxite):
    betagp = torch.dot(gk, pk) * beta
    if fun(xk + alpha * pk) > fk + alpha * betagp:
        return backwardArmijo(fun, xk, fk, gk, alpha, pk, beta, rho, maxite)
    else:
        j = 0
        while fun(xk + alpha * pk) <= fk + alpha * betagp and j < maxite:
            alpha /= rho
            j += 1
        if j >= maxite:
            print("linesearch max exceeded!")
        return rho * alpha, j
        
def backForwardArmijo(fun, xk, fk, gk, alpha, pk, beta, rho, maxite):
    betagp = torch.dot(gk, pk) * beta
    assert betagp < 0
    f = fun(xk + alpha * pk)
    if f > fk + alpha * betagp:
        return backwardArmijo(fun, xk, fk, gk, alpha, pk, beta, rho, maxite)
    else:
        j = 1
        while f <= fk + alpha * betagp and j < maxite:
            alpha /= rho
            fn = fun(xk + alpha * pk)
            j += 1
            
            # check for maximum descent
            if f < fn:
                break
            
            f = fn
        if j >= maxite:
            print("linesearch max exceeded!")
        return rho * alpha, j