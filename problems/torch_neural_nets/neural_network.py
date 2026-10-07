# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:49:54 2024

@author: uqalim8
"""

import torch.nn as nn

class auto_Encoder_MNIST(nn.Module):
    def __init__(self):
        super(auto_Encoder_MNIST, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 512),
                                      nn.Tanh(),
                                      nn.Linear(512, 256),
                                      nn.Tanh(),
                                      nn.Linear(256, 128),
                                      nn.Tanh(),
                                      nn.Linear(128, 64),
                                      nn.Tanh(),
                                      nn.Linear(64, 32),
                                      nn.Tanh(),
                                      nn.Linear(32, 16),
                                      )
        self.decoder = nn.Sequential(nn.Linear(16, 32),
                                      nn.Tanh(),
                                      nn.Linear(32, 64),
                                      nn.Tanh(),
                                      nn.Linear(64, 128),
                                      nn.Tanh(),
                                      nn.Linear(128, 256),
                                      nn.Tanh(),
                                      nn.Linear(256, 512),
                                      nn.Tanh(),
                                      nn.Linear(512, 28*28),
                                      )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class FFN(nn.Module):
    
    """
    Do not initialise the weights at zeros
    """
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._nn = nn.Sequential(nn.Linear(input_dim, 128),
                                 #nn.Tanh(),
                                 #nn.Linear(256, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, output_dim),
                                 nn.Softmax(dim = 1))

    def forward(self, x):
        return self._nn(x)
    
