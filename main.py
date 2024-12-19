# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:44:20 2024

@author: BinuJoseA
"""

import random
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from incdbscan import incdbscanner
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import incdbscan as f1
from scipy.spatial import distance
import dunn as f2
import davis as f3
import score as f4

population = [[random.uniform(min_eps, max_eps), random.randint(min_min_samples, max_min_samples)] for _ in range(population_size)]
print('population =',population)   
for gen in range(generations):
    # Compute the objective functions for the current population
    objectives = []
    for individual in population:
        eps, min_samples = individual
        dunn, davis, randindex = incremental_dbscan(eps, min_samples)  # You need to implement this function
        objectives.append([dunn, davis, randindex])
    print('objectives =',objectives)
    
    # Non-dominated sorting to find Pareto fronts
    fronts = non_dominated_sort(objectives)  # You can use the non_dominated_sort function from the previous answer
    print('fronts =',fronts)
    # Crowding distance calculation for diversity
    
    crowding_distances = crowding_distance(objectives, fronts)
    print('crowding_distances =',crowding_distances)
    
    # Create a new population through selection, crossover, and mutation
    new_population = []
    while len(new_population) < population_size:
        
        parent1, parent2 = tournament_selection(population, objectives, fronts, crowding_distances)
        print('parent1=',parent1)
        print('parent2=',parent2)
        
        child = crossover(parent1, parent2, crossover_rate)
        print('child=',child)
        
        child = mutate(child, mutation_rate) 
        
        new_population.append(child)
    
    # Replace the old population with the new population
    population = new_population
    
# After all generations, return the Pareto-optimal solutions
print('population =',population)
objectives = []
for individual in population[:5]:
    eps, min_samples,mergthld = individual
    dunn, davis, randindex,scoreindex = incremental_dbscan(eps, min_samples,mergthld)  # You need to implement this function
    objectives.append([dunn, davis, randindex,scoreindex])

# Sort the population based on both objective values (Dunn Index and Davies-Bouldin Index)
b_sorted_population_and_objectives = sorted(zip(objectives, population), key=lambda x: (x[1][1], x[0][2]))

# Extract the best solutions (the first 5 in this case)
import matplotlib.pyplot as plt

best_solutions = [sol[0] for sol in b_sorted_population_and_objectives[:5]]
best_parameters = [sol[1] for sol in b_sorted_population_and_objectives[:5]]
