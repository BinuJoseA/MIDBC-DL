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

def crowding_distance(objectives, fronts):
    crowding_distances = [0] * len(objectives)
    
    for front in fronts:
        num_objectives = len(objectives[0])
        front_with_objectives = [(i, objectives[i]) for i in front]  # Pair each solution with its objectives
        #print('front_with_objectives=',front_with_objectives)
        
        # Calculate crowding distance for each objective separately
        for obj_index in range(num_objectives):
            front_with_objectives.sort(key=lambda x: x[1][obj_index])  # Sort by the objective
            
            # Set crowding distance for boundary solutions to infinity
            crowding_distances[front_with_objectives[0][0]] = float('inf')
            crowding_distances[front_with_objectives[-1][0]] = float('inf')
            #print('front_with_objectives[-1][1][obj_index]=',front_with_objectives[-1][1][obj_index])
            #print('front_with_objectives[0][1][obj_index]=',front_with_objectives[0][1][obj_index])
            
            # Calculate crowding distance for other solutions
            for i in range(1, len(front_with_objectives) - 1):
                if front_with_objectives[-1][1][obj_index] != front_with_objectives[0][1][obj_index]:
                    crowding_distances[front_with_objectives[i][0]] += (front_with_objectives[i + 1][1][obj_index] - front_with_objectives[i - 1][1][obj_index]) / (front_with_objectives[-1][1][obj_index] - front_with_objectives[0][1][obj_index])
    
    return crowding_distances
