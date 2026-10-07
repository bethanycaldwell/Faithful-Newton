# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:20:15 2022

@author: uqalim8
"""

import torch, numpy, pandas, sklearn
import sklearn.datasets as skdatasets
import torchvision.datasets as datasets
from hyperparameters import cTYPE, cCUDA
from torchvision.transforms import v2

TEXT = "{:<20} : {:>20}"

def prepareData(dataset, one_hot):
        
    if dataset == "MNISTb":
        print(TEXT.format("Dataset", dataset))
        return MNIST(one_hot, 2)
    
    if dataset == "MNIST":
        print(TEXT.format("Dataset", dataset))
        return MNIST(one_hot, 10)
    
    if dataset == "CIFAR10b":
        print(TEXT.format("Dataset", dataset))
        return CIFAR10(one_hot, 2)
    
    if dataset == "CIFAR10":
        print(TEXT.format("Dataset", dataset))
        return CIFAR10(one_hot, 10)
    
    if dataset == "MNISTs":
        print(TEXT.format("Dataset", dataset))
        return MNISTs(one_hot, 10)
    
    if dataset == "MNISTsb":
        print(TEXT.format("Dataset", dataset))
        return MNISTs(one_hot, 2)
    
    if dataset == "DelhiClimate":
        print(TEXT.format("Dataset", dataset))
        return DelhiClimate()
    
    if dataset == "Ethylene":
        print(TEXT.format("Dataset", dataset))
        return Ethylene()
    
    if dataset == "Covtype":
        print(TEXT.format("Dataset", dataset))
        return Covtype(one_hot, 7)
        
    if dataset == "DTD":
        print(TEXT.format("Dataset", dataset))
        return DTD(one_hot, 47)
        
def MNISTs(one_hot, classes):
    """
    MNIST small size (8 by 8 pixels)

    """
    trainX, trainY = skdatasets.load_digits(return_X_y = True)
    trainX = torch.tensor(trainX, dtype = cTYPE, device = cCUDA)
    trainY = torch.tensor(trainY, dtype = torch.long, device = cCUDA) % classes
    
    if one_hot:
        trainY = torch.nn.functional.one_hot(trainY, classes).to(cTYPE)
        
    return trainX, trainY
    
def MNIST(one_hot, classes):
    
    train_set = datasets.MNIST("./", train = True, download = True)
    test_set = datasets.MNIST("./", train = False, download = True)
    
    X = torch.cat([torch.tensor(train_set.data.detach().reshape(train_set.data.shape[0], -1), 
                                dtype = cTYPE, device = cCUDA),
                   torch.tensor(test_set.data.detach().reshape(test_set.data.shape[0], -1), 
                                dtype = cTYPE, device = cCUDA)], dim = 0) 
    
    Y = torch.cat([torch.tensor(train_set.targets.detach(), device = cCUDA), 
                   torch.tensor(test_set.targets.detach(), device = cCUDA)], dim = 0) % classes
    
    del train_set, test_set

    if one_hot:
        Y = torch.nn.functional.one_hot(Y.long(), classes).to(cTYPE)
    
    print(TEXT.format("Data size", str(tuple(X.shape))))
    return X, Y
        
def CIFAR10(one_hot, classes):

    train_set = datasets.CIFAR10("./", train = True, download = True)
    test_set = datasets.CIFAR10("./", train = False, download = True)
    
    X = torch.cat([torch.tensor(train_set.data.reshape(train_set.data.shape[0], -1), 
                                dtype = cTYPE, device = cCUDA),
                   torch.tensor(test_set.data.reshape(test_set.data.shape[0], -1), 
                                dtype = cTYPE, device = cCUDA)], dim = 0)
    
    Y = torch.cat([torch.tensor(train_set.targets, device = cCUDA), 
                   torch.tensor(test_set.targets, device = cCUDA)], dim = 0) % classes
    
    del train_set, test_set
    
    if one_hot:
        Y = torch.nn.functional.one_hot(Y.long(), classes).to(cTYPE)
    
    print(TEXT.format("Data size", str(tuple(X.shape))))
    return X, Y

def Covtype(one_hot, classes):
    X, Y = sklearn.datasets.fetch_covtype(return_X_y = True)
    X = torch.tensor(X, dtype = cTYPE, device = cCUDA)
    Y = torch.tensor(Y, dtype = cTYPE, device = cCUDA) - 1
    Y = Y % classes
    
    if one_hot:
        Y = torch.nn.functional.one_hot(Y.long(), classes).to(cTYPE)
        
    print(TEXT.format("Data size", str(tuple(X.shape)))) 
    return X, Y
        
def DelhiClimate(window = 7):
    
    train = pandas.read_csv("./custom_data/DailyDelhiClimateTrain.csv")
    train = torch.tensor(train.drop("date", axis = 1).to_numpy(), dtype = cTYPE)
    
    #Standardise
    std, mean = torch.std_mean(train, dim = 0)
    train = (train - mean) / std
    
    n, d = train.shape
    trainX = torch.zeros((n - window, window, d), dtype = cTYPE)
    trainY = torch.zeros((n - window, d), dtype = cTYPE)
    
    for i in range(window, n):
        trainX[i - window] = train[i - window : i]
        trainY[i - window] = train[i]
        
    return trainX, trainY, None, None
        
def Ethylene(window = 2, stride = 1):
    classA = torch.tensor(numpy.loadtxt("./custom_data/mean_ethylene_CO.txt", delimiter = ","))
    classB = torch.tensor(numpy.loadtxt("./custom_data/mean_ethylene_methane.txt", delimiter = ","))

    prepX = torch.concat([classA, classB], dim = 0)
    #std, mean = torch.std_mean(prepX, dim = 0)
    #prepX = (prepX - mean) / std
    
    n, d = classA.shape
    m, _ = classB.shape
    del classA, classB
    
    trainX = torch.zeros((len(range(window, n, stride)) +
                          len(range(window, m, stride)), window, d + 1), dtype = cTYPE)
        
    trainY = torch.zeros((len(range(window, n, stride)) + 
                          len(range(window, m, stride)), 16), dtype = cTYPE)
    j = 0
    for i in range(window, n, stride):
        trainX[j, :, 1:] = prepX[i - window : i]
        trainY[j] = prepX[i, 2:]
        j += 1

    for i in range(window, m, stride):
        trainX[j, :, 1:] = prepX[n + i - window : n + i]
        trainX[j, :, 0] = 1.
        trainY[j] = prepX[n + i, 2:]
        j += 1
        
    #shuffle
    n = torch.randperm(trainX.shape[0])
    print(TEXT.format("Data size", str(tuple(trainX.shape))))
    return trainX[n], trainY[n], None, None

def DTD(one_hot, classes):
    trans = v2.Compose([v2.Resize(150), v2.CenterCrop(150), v2.PILToTensor()])
    train_set = datasets.DTD("./", split = "train", transform = trans, download = True)
    #test_set = datasets.DTD("./", train = False, transform = v2.Resize(150, 150), download = True)
    
    n = len(train_set)
    X = torch.zeros((n, 150*150*3), dtype=cTYPE)
    Y = torch.zeros(n, dtype=torch.long)
    
    i = 0
    for img, label in train_set:
        X[i] = img.reshape(-1) / 255.   # C,H,W -> H,W,C
        Y[i] = label
        i += 1
        
    del train_set    
    
    X = X.to(cCUDA)
    Y = Y.to(cCUDA)
    
    #X = torch.tensor(train_set.data.reshape(train_set.data.shape[0], -1), dtype = cTYPE, device = cCUDA)
    #X = torch.cat([torch.tensor(train_set.data.reshape(train_set.data.shape[0], -1), 
    #                            dtype = cTYPE, device = cCUDA),
    #               torch.tensor(test_set.data.reshape(test_set.data.shape[0], -1), 
    #                            dtype = cTYPE, device = cCUDA)], dim = 0)
    #Y = torch.tensor(train_set.targets, device = cCUDA)
    #Y = torch.cat([torch.tensor(train_set.targets, device = cCUDA), 
    #               torch.tensor(test_set.targets, device = cCUDA)], dim = 0) % classes
    
    #del train_set, test_set
    
    if one_hot:
        Y = torch.nn.functional.one_hot(Y.long(), classes).to(cTYPE)
    
    print(TEXT.format("Data size", str(tuple(X.shape))))
    return X, Y