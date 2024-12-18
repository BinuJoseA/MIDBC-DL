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

def tournament_selection(population, objectives, fronts, crowding_distances, tournament_size=2):
    selected_parents = []
    #iterations = 0
    
    while len(selected_parents) < 2 :  # You can adjust the number of parents needed
        # Randomly select individuals for the tournament
        tournament_individual_indices = random.sample(range(len(population)), tournament_size)
        #print('tournament_individual_indices=',tournament_individual_indices)
        if len(tournament_individual_indices) < 2:
            continue
        
        rank_dict = {}
        for rank, sublist in enumerate(fronts):
            for element in sublist:
                rank_dict[element] = rank
                
        # Calculate the rank and crowding distance of each individual in the tournament
        tournament_ranks = [rank_dict[i] for i in tournament_individual_indices]
        #print('tournament_ranks=',tournament_ranks)
        tournament_crowding_distances = [crowding_distances[i] for i in tournament_individual_indices]
        #print('tournament_crowding_distances=',tournament_crowding_distances)
        
        
        # Find the best individual in the tournament based on rank and crowding distance
        best_individual_index = None
        for i in range(tournament_size):
            if best_individual_index is None or tournament_ranks[i] < tournament_ranks[best_individual_index] or (tournament_ranks[i] == tournament_ranks[best_individual_index] and tournament_crowding_distances[i] > tournament_crowding_distances[best_individual_index]):
                best_individual_index = i
        
        #print('best_individual_index=',best_individual_index)
        #print('population[tournament_individual_indices[best_individual_index]]=',population[tournament_individual_indices[best_individual_index]])
        # Add the best individual to the selected parents
        selected_parents.append(population[tournament_individual_indices[best_individual_index]])
    
        #iterations += 1
        
    return selected_parents
