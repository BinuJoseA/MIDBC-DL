# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:50:36 2023

@author: BinuJoseA
"""

import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances

def euclidean_distance(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))

def calculate_centroid(points):
    return np.mean(points, axis=0)

def calculate_dunn_index(data, labels):
    labels_array = np.array(labels)
    unique_labels = np.unique(labels_array[labels_array != -1])
    print('unique_labels=',unique_labels)

    num_clusters = len(unique_labels)
    
    if num_clusters < 2:
        return 0.0  # Dunn Index is not defined for less than two clusters
    
    # Convert data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Calculate the inter-cluster distances
    inter_cluster_distances = []
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            indices_i = np.where(labels == unique_labels[i])[0]
            indices_j = np.where(labels == unique_labels[j])[0]
            distance = euclidean_distance(calculate_centroid(data[indices_i]), calculate_centroid(data[indices_j]))
            inter_cluster_distances.append(distance)

    # Calculate the minimum inter-cluster distance
    min_inter_cluster_distance = np.min(inter_cluster_distances)

    # Calculate the intra-cluster distances
    intra_cluster_distances = []
    for i in range(num_clusters):
        indices_i = np.where(labels == unique_labels[i])[0]
        distances = [euclidean_distance(data[j], calculate_centroid(data[indices_i])) for j in indices_i]
        intra_cluster_distances.append(np.max(distances))

    # Calculate the maximum intra-cluster distance
    max_intra_cluster_distance = np.max(intra_cluster_distances)
    
    # Check if max_intra_cluster_distance is zero
    if max_intra_cluster_distance == 0:
        return float('inf')  # Return infinity when max_intra_cluster_distance is zero

    # Calculate the Dunn Index
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance

    return dunn_index

