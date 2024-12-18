# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:50:36 2024

@author: BinuJoseA
"""

import numpy as np
from scipy.spatial.distance import cdist

def compute_SI(clusters):
    """
    Compute the Separation Index (SI) for the given clusters.

    Parameters:
    clusters: list of numpy arrays
        Each element in the list represents a cluster, which is a numpy array of points.

    Returns:
    float
        The computed SI value.
    """
    num_clusters = len(clusters)
    if num_clusters < 2:
        raise ValueError("There must be at least two clusters to compute SI.")

    # Compute P: Minimum inter-cluster distance
    inter_cluster_distances = []
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            dist_ij = cdist(clusters[i], clusters[j], metric='euclidean')
            inter_cluster_distances.append(np.min(dist_ij))
    P = np.min(inter_cluster_distances)

    # Compute Q: Maximum width of any cluster
    cluster_widths = []
    for cluster in clusters:
        width = np.max(cdist(cluster, cluster, metric='euclidean'))
        cluster_widths.append(width)
    Q = np.max(cluster_widths)

    # Compute R: Sum of intra-cluster compactness terms
    R = 0
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j:
                cluster_i = clusters[i]
                cluster_j = clusters[j]

                # Centroids of clusters i and j
                centroid_i = np.mean(cluster_i, axis=0)
                centroid_j = np.mean(cluster_j, axis=0)

                # Compute intra-cluster distances
                intra_i = np.sum(cdist(cluster_i, [centroid_i], metric='euclidean'))
                intra_j = np.sum(cdist(cluster_j, [centroid_j], metric='euclidean'))

                # Compute inter-cluster distance
                inter_ij = np.min(cdist(cluster_i, cluster_j, metric='euclidean'))

                R += np.max((intra_i + intra_j) / inter_ij)

    # Compute S: Total number of clusters
    S = num_clusters

    # Compute SI
    PS = P * S
    QR = Q * R

    SI = (PS - QR) / (Q * S)
    return SI
