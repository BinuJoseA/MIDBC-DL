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

def non_dominated_sort(objectives):
    # Initialize dominance counts and domination list for each individual
    n = len(objectives)
    domination_count = [0] * n
    dominated_list = [[] for _ in range(n)]
    
    # Create a list to store the Pareto fronts
    fronts = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                domination_count[j] += 1
                dominated_list[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                domination_count[i] += 1
                dominated_list[j].append(i)
    
    # Find the individuals in the first Pareto front
    front1 = [i for i in range(n) if domination_count[i] == 0]
    fronts.append(front1)
    
    # Iterate to find subsequent fronts
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for ind in fronts[i]:
            for dominated in dominated_list[ind]:
                domination_count[dominated] -= 1
                if domination_count[dominated] == 0:
                    next_front.append(dominated)
        i += 1
        fronts.append(next_front)
    
    # Remove empty fronts
    fronts = [front for front in fronts if front]
    
    return fronts

def dominates(objective1, objective2):
    # Check if objective1 dominates objective2
    # Returns True if objective1 dominates, False otherwise
    for i in range(len(objective1)):
        if objective1[i] < objective2[i]:
            return False
    return any(objective1[j] > objective2[j] for j in range(len(objective1)))
