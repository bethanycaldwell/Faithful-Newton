# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:54:40 2024

@author: uqalim8
"""
from .functions.softmax import softmax
from .datasets import prepareData
from .regularizers import initReg
from .torch_neural_nets.neural_network import FFN
from .torch_neural_nets.neural_utils import nnWrapper
import torch.nn as nn
TEXT = "{:<20} : {:>20}"

def problems(problems_type, lamb):
    
    func, reg, dataset = problems_type.split("_")
    lamb = 0 if lamb == "None" else float(lamb)
    if func == "softmax":
        reg = initReg(reg, lamb)
        assert dataset in ["MNIST", "CIFAR10", "Covtype", "DTD"]
        X, Y = prepareData(dataset, True)
        print(TEXT.format("No. Samples", X.shape[0]))
        print(TEXT.format("Dimensions", X.shape[-1] * Y.shape[-1]))
        return X.shape[-1] * Y.shape[-1], lambda w, v : softmax(X, Y, w, v, reg)
    
    if func == "FFN-MSELoss":
        reg = initReg(reg, lamb)
        assert dataset in ["MNIST", "CIFAR10", "Covtype",
                           "MNISTb", "CIFAR10b", "MNISTs", "MNISTsb"]
        X, Y = prepareData(dataset, True)
        d, c = X.shape[-1], Y.shape[-1]
        ffn, loss = FFN(d, c), nn.MSELoss()
        f = nnWrapper(ffn, loss, reg, X, Y)
        print(TEXT.format("No. Samples", X.shape[0]))
        print(TEXT.format("Dimensions", f.size))
        return f.size, f
    
    raise ValueError("Problem type should follow valid, func_reg_dataset, arguments")
    
    
