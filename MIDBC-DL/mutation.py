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

def mutate(child, mutation_rate):
    mutated_child = []
    
    for gene in child:
        if random.random() < mutation_rate:
            # Apply Random-mutation by adding a small random value to the gene
            mutation = random.uniform(-0.1, 0.1)  # You can adjust the mutation range
            mutated_gene = gene + mutation
        else:
            mutated_gene = gene
        
        mutated_child.append(mutated_gene)
    
    return mutated_child
