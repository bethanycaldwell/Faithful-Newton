# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:54:40 2024

@author: uqalim8
"""
import torch
from hyperparameters import cMCRNPC
from .functions.softmax import softmax
from .functions.softmax_cubic import softmax_cubic
from .datasets import prepareData
from .regularizers import initReg
from .torch_neural_nets.neural_network import FFN
from .torch_neural_nets.neural_utils import nnWrapper
import torch.nn as nn
TEXT = "{:<20} : {:>20}"

def problems(problems_type, lamb):
    """
    Given a problem type and a regularization parameter, this function prepares the corresponding optimization problem.
    The problem type is expected to be in the format "func_reg_dataset", where:
    - func is the objective function
    - reg is the regularization type
    - dataset is the dataset name

    Parameters
    ----------
    problems_type : str
        The type of the optimization problem.
    lamb : float or str
        The regularization parameter.

    Returns
    -------
    tuple
        A tuple containing the size of the optimization problem and the objective function.
    """
    func, reg, dataset = problems_type.split("_")
    lamb = 0 if lamb == "None" else float(lamb)

    if func == "softmax":
        reg = initReg(reg, lamb)
        assert dataset in ["MNIST", "MNISTb", "MNISTsb", "MNISTs", "CIFAR10b", "CIFAR10", "CIFAR100", "DTD", "Covtype"]
        X, Y = prepareData(dataset, True)
        print(TEXT.format("No. Samples", X.shape[0]))
        print(TEXT.format("Dimensions", X.shape[-1] * Y.shape[-1]))
        return X.shape[-1] * Y.shape[-1], lambda w, v : softmax(X, Y, w, v, reg)
    
    if func == "FFN-MSELoss":
        reg = initReg(reg, lamb)
        assert dataset in ["MNIST", "MNISTb", "MNISTsb", "MNISTs", "CIFAR10b", "CIFAR10", "CIFAR100", "DTD", "Covtype"]
        X, Y = prepareData(dataset, True)
        d, c = X.shape[-1], Y.shape[-1]
        ffn, loss = FFN(d, c), nn.MSELoss()
        f = nnWrapper(ffn, loss, reg, X, Y)
        print(TEXT.format("No. Samples", X.shape[0]))
        print(TEXT.format("Dimensions", f.size))
        return f.size, f
    
    if func == "softmax-cubic":
        reg = initReg(reg, lamb)
        assert dataset in ["MNIST", "MNISTb", "MNISTsb", "MNISTs", "CIFAR10b", "CIFAR10", "CIFAR100", "DTD", "Covtype"]
        X, Y = prepareData(dataset, True)
        print(TEXT.format("No. Samples", X.shape[0]))
        print(TEXT.format("Dimensions", X.shape[-1] * Y.shape[-1]))
        return X.shape[-1] * Y.shape[-1], lambda w, v : softmax_cubic(X, Y, w, v, reg, M = cMCRNPC.M0)
        
    raise ValueError("Problem type should follow valid, func_reg_dataset, arguments")
