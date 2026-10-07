# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:56:23 2023

@author: uqalim8
"""

import matplotlib.pyplot as plt
import os, json, torch

FOLDER_PATH = "./cifar10_results/"
SPLIT = ".json"

def keys(x):
    n = x.split("%")[0]
    if x == "Single Sample.json":
        return 0
    else:
        return float(n)
    
def drawPlots(records, stats, folder_path):
    STATS = {"ite":"Iterations", "inite":"Inner Iteration", "orcs":"Oracle Calls",
             "time":"Time(second)", "f":"Function Value", "g_norm":"Norm of Gradient",
             "alpha":"Step Size", "acc":"Accuracy"}
    for x, y in stats:
        for name, record in records:
            plt.semilogx(torch.tensor(record[x]) + 1, record[y], label = name)
        plt.xlabel(STATS[x])
        plt.ylabel(STATS[y])
        plt.legend()
        plt.savefig(FOLDER_PATH + f"{x}_{y}.png")
        plt.close()
    
if __name__ == "__main__":
    
    files = os.listdir(FOLDER_PATH)
    files = filter(lambda x : ".json" in x, files)
    #files.sort(key = keys)
    records = []
    for i in files:
        name, _ = i.split(SPLIT)
        with open(FOLDER_PATH + i, "r") as f:
            records.append((name, json.load(f)))
                    
    drawPlots(records, (("orcs", "f"), ("ite", "f"), ("orcs", "g_norm"), ("ite", "g_norm")), FOLDER_PATH)
        