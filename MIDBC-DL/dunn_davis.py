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

'''
data_points = [
    [0, 0], [1, 0], [1, 1], [2, 2], [0, 1],
    [3, 1], [3, 0], [3, 2], [6, 3], [0, 2],
    [1, 2], [6, 4], [6, 5]
]

# Sample clusters (list of data points)
clist = [
    [[0, 0], [1, 0], [1, 1], [0, 1], [0, 2], [3, 1], [3, 0], [3, 2], [2, 2], [1, 2]],
    [[6, 4], [6, 3], [6, 5]]
]

data_point_labels = []

# Iterate through the data points in the dataset
for data_point_index, data_point in enumerate(data_points):
    found = False  # Flag to check if the data point is found in any cluster
    for cluster_index, cluster1 in enumerate(clist):
        if data_point in cluster1:
            data_point_labels.append(cluster_index)  # Assign the cluster index as the label
            found = True
            break
    if not found:
        data_point_labels.append(-1) 

print('data_point_labels=',data_point_labels)
db= calculate_davies_bouldin_index(data_points, data_point_labels)
print('db=',db)
dn=calculate_dunn_index(data_points, data_point_labels)
print('dn=',dn)
#data_points = np.array(data_points)
db_score = davies_bouldin_score(data_points, data_point_labels)
print('db_score=',db_score)
'''