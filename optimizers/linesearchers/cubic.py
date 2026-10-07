# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:15:49 2022

@author: uqalim8
"""

def dampedNewtonCGLinesearch(fun, xk, fk, alpha, pk, normpk, beta, rho, maxite):
    const = beta * normpk / 6
    j = 1
    while fun(xk + alpha * pk) > fk - const * (alpha ** 3) and j < maxite:
        alpha *= rho
        j += 1
    if j >= maxite:
        print("linesearch max exceeded!")
    return alpha, j

def dampedNewtonCGbackForwardLS(fun, xk, fk, alpha, pk, normpk, beta, rho, maxite):
    const = beta * normpk / 6
    if fun(xk + alpha * pk) > fk - const * (alpha ** 3):
        return dampedNewtonCGLinesearch(fun, xk, fk, alpha, pk, normpk, beta, rho, maxite)
    else:
        j = 1
        while fun(xk + alpha * pk) <= fk - const * (alpha ** 3) and j < maxite:
            alpha = 2 * alpha
            j += 1
        if j >= maxite:
            print("linesearch max exceeded!")
        return alpha / 2, j
