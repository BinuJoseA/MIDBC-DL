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

def calculate_davies_bouldin_index(data, labels):
    labels_array = np.array(labels)
    unique_labels = np.unique(labels_array[labels_array != -1])
    num_clusters = len(unique_labels)
    
    if num_clusters < 2:
        return 0.0  # Davies-Bouldin Index is not defined for less than two clusters
    
    # Convert data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels, dtype=int)

    # Calculate the centroids of each cluster
    centroids = [calculate_centroid(data[labels == label]) for label in unique_labels]

    # Calculate the average distance within each cluster
    avg_distances = []
    for i in range(num_clusters):
        distances = [euclidean_distance(data[j], centroids[i]) for j in np.where(labels == unique_labels[i])[0]]
        avg_distances.append(np.mean(distances))

    # Calculate the average distance between clusters
    # Calculate the similarity between cluster i and cluster j
    avg_inter_distances = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            distance = euclidean_distance(centroids[i], centroids[j])
            avg_inter_distances[i, j] = distance
            avg_inter_distances[j, i] = distance

    # Calculate the Davies-Bouldin Index
    db_index = 0
    for i in range(num_clusters):
        max_val = -np.inf
        for j in range(num_clusters):
            if i != j:
                val = (avg_distances[i] + avg_distances[j]) / avg_inter_distances[i, j]
                max_val = max(max_val, val)
        db_index += max_val

    # The Davies-Bouldin Index is the average of the maximum similarities for each cluster    
    db_index /= num_clusters

    return db_index
