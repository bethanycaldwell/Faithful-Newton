# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:12:13 2022

@author: uqalim8

Example command line: python -m cProfile -o program.prof main.py ./Results CRN
or python main.py ./Results CRN
"""

from utils import execute, saveRecords
import hyperparameters as hyper 
import sys

SEED = 2025 # integers
FOLDER_PATH = f"./{sys.argv[1]}"
PROBLEM_TYPE = "softmax_2-norm_MNISTsb" # functions: [softmax, FFN-MSELoss, softmax-cubic], 
                                        # regularizers: [None, non-convex, 2-norm, LASSO], 
                                        # datasets: [MNIST, MNISTb, MNISTsb, MNISTs, CIFAR10b, CIFAR10, CIFAR100, DTD, Covtype]
LAMBDA = 0.1
INITX0 = "uniform" #zeros, ones, uniform, normal, torch
VERBOSE = 1

algorithm_name = sys.argv[2]

algorithms = {
    "NewtonMR-NC": hyper.cMCRNPC,
    "NewtonCR-NC": hyper.cMCRNPC,
    "NewtonCR": hyper.cNWL,
    "NewtonCG": hyper.cNWL,
    "FaithfulNewtonCR": hyper.cFN,
    "FaithfulNewtonCR-reg": hyper.cFN,
    "L-BFGS": hyper.cL_BFGS,
    "TR_Steihaug": hyper.cTR_STEI,
    "AdaN+": hyper.cAdaN,
    "AdaN": hyper.cAdaN,
    "CRN": hyper.cCubic,
    "GLRT": hyper.GLRT,
    "AccCRN": hyper.cCubic,
    "NATA": hyper.cNATA,
    "OptMS": hyper.cMS,
    "linesearchGD": hyper.cGD
}

if __name__ == "__main__":
    print("\n" + 21 * "**" + "\n" + 15 * " " + "Running on", str(hyper.cCUDA), "\n" + 21 * "**")
    algo = execute(FOLDER_PATH, PROBLEM_TYPE, LAMBDA, algorithm_name, INITX0, algorithms[algorithm_name], VERBOSE, SEED)
    saveRecords(FOLDER_PATH, PROBLEM_TYPE, algorithm_name, algo.record)
