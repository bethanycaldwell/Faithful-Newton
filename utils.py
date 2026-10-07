# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:12:13 2022

@author: uqalim8
"""

import os, torch, json
from optimizers.optimizer_utils import init_algorithms
from problems.problems import problems

from hyperparameters import cTYPE, cCUDA

TEXT = "{:<20} : {:>20}"

def makeFolder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

def saveRecords(folder_path, alg, file):
    if folder_path[-1] != "/":
        folder_path += "/"
    folder_path += f"{alg}.json"
    with open(folder_path, "w") as f:
        json.dump(file, f)
        
def openRecords(folder_path, dataset, func):
    if folder_path[-1] != "/":
        folder_path += "/"
    files = os.listdir(f"{folder_path}{dataset}_{func}/")
    records = []
    for i in files:
        with open(folder_path + i, "r") as f:
            records.append((i, json.load(f)))
    return records
                
def initx0(x0_type, size):
    
    if not type(x0_type) == str:
        print(TEXT.format("x0", "initialised"))
        return x0_type.to(cCUDA)
    
    if x0_type == "ones":
        print(TEXT.format("x0", x0_type))
        return torch.ones(size, dtype = cTYPE, device = cCUDA) * 1
    
    if x0_type == "zeros":
        print(TEXT.format("x0", x0_type))
        return torch.zeros(size, dtype = cTYPE, device = cCUDA)
    
    if x0_type == "normal":
        print(TEXT.format("x0", x0_type))
        return torch.randn(size, dtype = cTYPE, device = cCUDA) * 0.1
    
    if x0_type == "uniform":
        print(TEXT.format("x0", x0_type))
        return torch.rand(size, dtype = cTYPE, device = cCUDA) * 0.01
    
def execute(folder_path, problem_type, lamb, algo, x0, const, verbose, seed):
    
    if type(seed) == int:
        torch.manual_seed(seed)
        
    makeFolder(folder_path)
    size, func = problems(problem_type, lamb)
    x0 = initx0(x0, size)
    
    # optimizing
    algo = init_algorithms(func, x0.clone(), algo, const)
    algo.optimize(verbose, lambda x : 0)
    return algo
