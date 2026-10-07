# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:34:26 2024

@author: uqalim8
"""
import torch

def CappedCG(H, b, zeta, epsilon, maxiter, M=0):
    g = -b
    y =  torch.zeros_like(g)
    kappa, tzeta, tau, T = para(M, epsilon, zeta)
    tHy = y.clone()
    tHY = y.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    r = g
    p = -g
    tHp = Avec(H, p) + 2*epsilon*p
    j = 1
    ptHp = torch.dot(p, tHp)
    norm_g = torch.norm(g)
    norm_p = norm_g
    rr = torch.dot(r, r)
    dType = 'Sol'
    relres = 1
    if ptHp < epsilon*norm_p**2:
        d = p
        dType = 'NC'
        return d, dType, j, ptHp, 1
    norm_Hp = torch.norm(tHp - 2*epsilon*p)
    if norm_Hp > M*norm_p:
        M = norm_Hp/norm_p
        kappa, tzeta, tau, T = para(M, epsilon, zeta)    
    while j < maxiter:
        alpha = rr/ptHp
        y = y + alpha*p
        #Y = torch.cat((Y, y.reshape(-1, 1)), 1) #record y
        norm_y = torch.norm(y)
        tHy = tHy + alpha*tHp
        #tHY = torch.cat((tHY, tHy.reshape(-1, 1)), 1) # record tHy
        norm_Hy = torch.norm(tHy - 2*epsilon*y)
        r = r + alpha*tHp
        rr_new = torch.dot(r, r) 
        beta = rr_new/rr
        rr = rr_new
        p = -r + beta*p #calculate Hr
        norm_p = torch.norm(p)    
        tHp_new = Avec(H, p) + 2*epsilon*p #the only Hessian-vector product
        j = j + 1
        tHr = beta*tHp - tHp_new #calculate Hr
        tHp = tHp_new
        norm_Hp = torch.norm(tHp - 2*epsilon*p)
        ptHp = torch.dot(p, tHp)  
        if  norm_Hp> M*norm_p:
            M = norm_Hp/norm_p
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        if norm_Hy > M*norm_y:
            M = norm_Hy/norm_y
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        norm_r = torch.norm(r)
        relres = norm_r/norm_g
#        print(norm_r/norm_g, tzeta)
        norm_Hr = torch.norm(tHr - 2*epsilon*r)
#        print(norm_r, torch.norm(H(y) + g))
        if  norm_Hr> M*norm_r:
            M = norm_Hr/norm_r         
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        if torch.dot(y, tHy) < epsilon*norm_y**2:
            d = y
            dType = 'NC'
            # print('y')
            return d, dType, j, torch.dot(y, tHy), relres
        elif norm_r < tzeta*norm_g:
            # print('relres', relres)
            d = y
            return d, dType, j, 0, relres
        elif torch.dot(p, tHp) < epsilon*norm_p**2:
            d = p
            dType = 'NC'
            # print('p')
            return d, dType, j, torch.dot(p, tHp), relres
        elif norm_r > torch.sqrt(T*tau**j)*norm_g:
            print('Uncomment tensors Y, tHY')
            alpha_new = rr/ptHp
            y_new = y + alpha_new*p            
            tHy_new = tHy + alpha_new*tHp
            for i in range(j):
                dy = y_new - Y[:, i]
                dtHy = tHy_new - tHY[:, i]
                if torch.dot(dy, dtHy) < epsilon*torch.norm(dy)**2:
                    d = dy
                    dType = 'NC'
                    print('dy')
                    return d, dType, j, torch.dot(dy, dtHy), relres
    print('Maximum iteration exceeded!')
    return y, dType, j, 0, relres

def para(M, epsilon, zeta):
    # if torch.tensor(M):
    #     M = M.item()
    kappa = (M + 2*epsilon)/epsilon
    tzeta = zeta/3/kappa
    # print('kappa', kappa)
    sqk = torch.sqrt(torch.tensor(float(kappa)))
    tau = sqk/(sqk + 1)
    T = 4*kappa**4/(1 + torch.sqrt(tau))**2
    return kappa, tzeta, tau, T

def Avec(A, x):
    if callable(A):
        return A(x)
    return torch.mv(A, x)