# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:50:36 2024

@author: BinuJoseA
"""

import random
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance

def crossover(parent1, parent2, crossover_rate):
    child = []
    #Apply Blend Crossover
    for i in range(len(parent1)):
        if random.random() < crossover_rate:
            # Calculate the blending range for crossover
            alpha = random.uniform(-0.5, 1.5)
            
            # Perform blend crossover for the current gene
            child_gene = (1 - alpha) * parent1[i] + alpha * parent2[i]
            child.append(child_gene)
        else:
            # If crossover doesn't occur, choose a gene from one of the parents randomly
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
    
    return child
