# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:46:34 2023

@author: BinuJoseA
"""



import random
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from incdbscannertest4 import incdbscanner
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import incdbscannertest4 as f1
from scipy.spatial import distance
import dunn as f2
import davis as f3
import score as f4
import dunn_davis as f2

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans



# Generate synthetic data
np.random.seed(42)
data = np.random.rand(300, 2)
true_labels = np.random.randint(0, 3, size=300)

# Clustering with pseudo-labeling
kmeans = KMeans(n_clusters=3)
kmeans_labels = kmeans.fit_predict(data)

# Create clusters from labels
clusters = [data[kmeans_labels == i] for i in np.unique(kmeans_labels)]

# Compute metrics
di = compute_DI(clusters)
dbi = compute_DBI(clusters)
ri = compute_RI(clusters, true_labels)
si = compute_SI(clusters)

# Input data for ANN
metrics = np.array([[di, dbi, ri, si]])

# Define ANN architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential([
    Input(shape=(4,)),  # Input layer
    Dense(64, activation='relu'),  # Hidden layer
    Dense(32, activation='relu'),  # Bottleneck layer
    Dense(1, activation='linear')  # Output layer: Score
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Simulate training data
# In practice, you would need a dataset of metrics and corresponding scores
X_train = np.random.rand(100, 4)  # Simulated DI, DBI, RI, SI values
y_train = np.random.rand(100, 1)  # Simulated scores

# Train the ANN
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# Predict scores for new metrics
predicted_scores = model.predict(metrics)

# Pseudo-labeling for clustering
def pseudo_label_clustering(data, scores):
    thresholds = np.percentile(scores, [25, 50, 75])
    labels = np.digitize(scores, bins=thresholds)
    return labels

# Generate pseudo-labels
pseudo_labels = pseudo_label_clustering(data, predicted_scores)

# Pareto front calculation
def calculate_pareto_front(scores):
    sorted_indices = np.argsort(scores, axis=0)
    pareto_ranks = np.zeros_like(scores, dtype=int)
    for i, idx in enumerate(sorted_indices):
        pareto_ranks[idx] = i + 1
    return pareto_ranks

pareto_fronts = calculate_pareto_front(predicted_scores)

print("Predicted Scores:", predicted_scores)
print("Pseudo-labels:", pseudo_labels)
print("Pareto Front Ranks:", pareto_fronts)
