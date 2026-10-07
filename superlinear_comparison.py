# -*- coding: utf-8 -*-
"""
Created on Sun May 18 19:44:12 2025

@author: uqalim8
"""

import torch, math
import matplotlib.pyplot as plt

Lg = 1e6
mu = 5
LH = 100
rho = 1/4
k = Lg / mu
D = 100
f0mfop = mu * D ** 2 / 2
fopt = 1e-14
N = 1e6
eps = 1e-6
arangeN = torch.arange(N)

assert LH * D / mu >= 694
assert Lg / mu >= 68

# # define C

locallower = 1e4 * mu ** 3 / (8 * LH ** 2)

C2 = max(LH * math.sqrt(f0mfop) / (math.sqrt(2) * rho * mu ** (3/2)),
         2 * LH * math.sqrt(k) * math.sqrt(f0mfop) / (3 * 1 * rho * (1 - 2 * rho) * mu ** (3/2)))
C0 = 1 - 1 / (1 + C2)     

GCk = C0 ** (arangeN / 2)
GSL = 1 - 1 / (1 + C2 * math.sqrt(f0mfop) * GCk)
cumGSL = torch.cat([torch.tensor(f0mfop).reshape(1), torch.cumprod(GSL, 0) * f0mfop])

lowernonlocalT = math.ceil(min(math.sqrt((Lg - mu) /(2 * mu)) / 14, (LH * D / (12 * mu)) ** (2/7) / (7 * 2 ** (1/7))))
lowerlocalT = math.log(math.log(mu ** 3 / (fopt * LH ** 2), 18),2)
print(lowernonlocalT + lowerlocalT)
print(len(cumGSL[cumGSL > fopt]))

# locallowerC = (LH ** 2) / ((108 * mu ) ** 2) * 18 ** (2 ** arangeN) 
# if len(lowerSeq[lowerSeq < locallower]):
#     locallowerSeq = lowerSeq[lowerSeq < locallower][0]
#     globallowerSeq = torch.cat([lowerSeq[lowerSeq >= locallower], torch.tensor(locallower).reshape(1)])
#     locallowerSeq = locallowerC[locallowerC >= eps]
    
#     lower = torch.cat([globallowerSeq.reshape(1), locallowerC])

# endClass = len(lower)
# plt.semilogy(arangeN[:endClass], lower, linestyle = "-.", color = "purple", label = "Linear")
# plt.axhline(locallowerC, linestyle = "--", color = "g", alpha = 0.25, label = "Local Region")
# plt.xlabel("iteration k", fontsize=12)
# #plt.ylabel(r"$Const(k) \cdot (f_0 - f^*)$", fontsize=17)
# plt.legend()
# plt.show()
