
import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances

def euclidean_distance(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))

def calculate_centroid(points):
    return np.mean(points, axis=0)


def min_distance_between_clusters(cluster1, cluster2):
    """Calculate the minimum distance between two clusters."""
    min_distance = float('inf')
    
    # Iterate through all pairs of points from each cluster
    for point1 in cluster1:
        for point2 in cluster2:
            # Calculate the distance between the current pair of points
            distance = euclidean_distance(point1, point2)
            # Update the minimum distance if needed
            if distance < min_distance:
                min_distance = distance
    
    return min_distance


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
    distance_app = []
    for i in range(num_clusters):
        distances = [euclidean_distance(data[j], centroids[i]) for j in np.where(labels == unique_labels[i])[0]]
        distance_app.append(np.sum(distances))
        avg_distances.append(np.mean(distances))

    # Calculate the average distance between clusters
    # Calculate the similarity between cluster i and cluster j
    '''
    avg_inter_distances = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            distance = euclidean_distance(centroids[i], centroids[j])
            avg_inter_distances[i, j] = distance
            avg_inter_distances[j, i] = distance
    '''        
    min_inter_distances = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            distance = min_distance_between_clusters(clist[i], clist[j])
            min_inter_distances[i, j] = distance
            min_inter_distances[j, i] = distance
    
    # Calculate the Davies-Bouldin Index
    db_index = 0
    for i in range(num_clusters):
        max_val = -np.inf
        for j in range(num_clusters):
            if i != j:
                #val = (avg_distances[i]/distance_app[i] + avg_distances[j]/distance_app[j]) / min_inter_distances[i, j]
                val = (avg_distances[i] + avg_distances[j]) / min_inter_distances[i, j]
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
    '''
    min_inter_distances = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            distance = min_distance_between_clusters(clist[i], clist[j])
            min_inter_distances[i, j] = distance
            min_inter_distances[j, i] = distance
    
    '''
    inter_cluster_distances = []
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            indices_i = np.where(labels == unique_labels[i])[0]
            indices_j = np.where(labels == unique_labels[j])[0]
            distance = euclidean_distance(calculate_centroid(data[indices_i]), calculate_centroid(data[indices_j]))
            inter_cluster_distances.append(distance)
    
    # Calculate the minimum inter-cluster distance
    min_inter_cluster_distance = np.min(inter_cluster_distances)
    #min_inter_cluster_distance = np.min(min_inter_distances)

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
'''
data_points = [[5.1,3.5,1.4,0.2],[4.9,3,1.4,0.2],[4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],[5,3.6,1.4,0.2],[5.4,3.9,1.7,0.4],[4.6,3.4,1.4,0.3],[5,3.4,1.5,0.2],[4.4,2.9,1.4,0.2],[4.9,3.1,1.5,0.1],[5.4,3.7,1.5,0.2],[4.8,3.4,1.6,0.2],[4.8,3,1.4,0.1],[4.3,3,1.1,0.1],[5.8,4,1.2,0.2],[5.7,4.4,1.5,0.4],[5.4,3.9,1.3,0.4],[5.1,3.5,1.4,0.3],[5.7,3.8,1.7,0.3],[5.1,3.8,1.5,0.3],[5.4,3.4,1.7,0.2],[5.1,3.7,1.5,0.4],[4.6,3.6,1,0.2],[5.1,3.3,1.7,0.5],[4.8,3.4,1.9,0.2],[5,3,1.6,0.2],[5,3.4,1.6,0.4],[5.2,3.5,1.5,0.2],[5.2,3.4,1.4,0.2],[4.7,3.2,1.6,0.2],[4.8,3.1,1.6,0.2],[5.4,3.4,1.5,0.4],[5.2,4.1,1.5,0.1],[5.5,4.2,1.4,0.2],[4.9,3.1,1.5,0.1],[5,3.2,1.2,0.2],[5.5,3.5,1.3,0.2],[4.9,3.1,1.5,0.1],[4.4,3,1.3,0.2],[5.1,3.4,1.5,0.2],[5,3.5,1.3,0.3],[4.5,2.3,1.3,0.3],[4.4,3.2,1.3,0.2],[5,3.5,1.6,0.6],[5.1,3.8,1.9,0.4],[4.8,3,1.4,0.3],[5.1,3.8,1.6,0.2],[4.6,3.2,1.4,0.2],[5.3,3.7,1.5,0.2],[5,3.3,1.4,0.2],[7,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],[5.5,2.3,4,1.3],[6.5,2.8,4.6,1.5],[5.7,2.8,4.5,1.3],[6.3,3.3,4.7,1.6],[4.9,2.4,3.3,1],[6.6,2.9,4.6,1.3],[5.2,2.7,3.9,1.4],[5,2,3.5,1],[5.9,3,4.2,1.5],[6,2.2,4,1],[6.1,2.9,4.7,1.4],[5.6,2.9,3.6,1.3],[6.7,3.1,4.4,1.4],[5.6,3,4.5,1.5],[5.8,2.7,4.1,1],[6.2,2.2,4.5,1.5],[5.6,2.5,3.9,1.1],[5.9,3.2,4.8,1.8],[6.1,2.8,4,1.3],[6.3,2.5,4.9,1.5],[6.1,2.8,4.7,1.2],[6.4,2.9,4.3,1.3],[6.6,3,4.4,1.4],[6.8,2.8,4.8,1.4],[6.7,3,5,1.7],[6,2.9,4.5,1.5],[5.7,2.6,3.5,1],[5.5,2.4,3.8,1.1],[5.5,2.4,3.7,1],[5.8,2.7,3.9,1.2],[6,2.7,5.1,1.6],[5.4,3,4.5,1.5],[6,3.4,4.5,1.6],[6.7,3.1,4.7,1.5],[6.3,2.3,4.4,1.3],[5.6,3,4.1,1.3],[5.5,2.5,4,1.3],[5.5,2.6,4.4,1.2],[6.1,3,4.6,1.4],[5.8,2.6,4,1.2],[5,2.3,3.3,1],[5.6,2.7,4.2,1.3],[5.7,3,4.2,1.2],[5.7,2.9,4.2,1.3],[6.2,2.9,4.3,1.3],[5.1,2.5,3,1.1],[5.7,2.8,4.1,1.3],[6.3,3.3,6,2.5],[5.8,2.7,5.1,1.9],[7.1,3,5.9,2.1],[6.3,2.9,5.6,1.8],[6.5,3,5.8,2.2],[7.6,3,6.6,2.1],[4.9,2.5,4.5,1.7],[7.3,2.9,6.3,1.8],[6.7,2.5,5.8,1.8],[7.2,3.6,6.1,2.5],[6.5,3.2,5.1,2],[6.4,2.7,5.3,1.9],[6.8,3,5.5,2.1],[5.7,2.5,5,2],[5.8,2.8,5.1,2.4],[6.4,3.2,5.3,2.3],[6.5,3,5.5,1.8],[7.7,3.8,6.7,2.2],[7.7,2.6,6.9,2.3],[6,2.2,5,1.5],[6.9,3.2,5.7,2.3],[5.6,2.8,4.9,2],[7.7,2.8,6.7,2],[6.3,2.7,4.9,1.8],[6.7,3.3,5.7,2.1],[7.2,3.2,6,1.8],[6.2,2.8,4.8,1.8],[6.1,3,4.9,1.8],[6.4,2.8,5.6,2.1],[7.2,3,5.8,1.6],[7.4,2.8,6.1,1.9],[7.9,3.8,6.4,2],[6.4,2.8,5.6,2.2],[6.3,2.8,5.1,1.5],[6.1,2.6,5.6,1.4],[7.7,3,6.1,2.3],[6.3,3.4,5.6,2.4],[6.4,3.1,5.5,1.8],[6,3,4.8,1.8],[6.9,3.1,5.4,2.1],[6.7,3.1,5.6,2.4],[6.9,3.1,5.1,2.3],[5.8,2.7,5.1,1.9],[6.8,3.2,5.9,2.3],[6.7,3.3,5.7,2.5],[6.7,3,5.2,2.3],[6.3,2.5,5,1.9],[6.5,3,5.2,2],[6.2,3.4,5.4,2.3],[5.9,3,5.1,1.8]]

# Sample clusters (list of data points)
'''
clist = [
    [[0, 0], [1, 0], [1, 1], [0, 1], [0, 2], [3, 1], [3, 0], [3, 2], [2, 2], [1, 2]],
    [[6, 4], [6, 3], [6, 5]]
]
'''
clist =[
        [[5.1,3.5,1.4,0.2],[4.9,3,1.4,0.2],[4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],[5,3.6,1.4,0.2],[5.4,3.9,1.7,0.4],[4.6,3.4,1.4,0.3],[5,3.4,1.5,0.2],[4.4,2.9,1.4,0.2],[4.9,3.1,1.5,0.1],[5.4,3.7,1.5,0.2],[4.8,3.4,1.6,0.2],[4.8,3,1.4,0.1],[4.3,3,1.1,0.1],[5.8,4,1.2,0.2],[5.7,4.4,1.5,0.4],[5.4,3.9,1.3,0.4],[5.1,3.5,1.4,0.3],[5.7,3.8,1.7,0.3],[5.1,3.8,1.5,0.3],[5.4,3.4,1.7,0.2],[5.1,3.7,1.5,0.4],[4.6,3.6,1,0.2],[5.1,3.3,1.7,0.5],[4.8,3.4,1.9,0.2],[5,3,1.6,0.2],[5,3.4,1.6,0.4],[5.2,3.5,1.5,0.2],[5.2,3.4,1.4,0.2],[4.7,3.2,1.6,0.2],[4.8,3.1,1.6,0.2],[5.4,3.4,1.5,0.4],[5.2,4.1,1.5,0.1],[5.5,4.2,1.4,0.2],[4.9,3.1,1.5,0.1],[5,3.2,1.2,0.2],[5.5,3.5,1.3,0.2],[4.9,3.1,1.5,0.1],[4.4,3,1.3,0.2],[5.1,3.4,1.5,0.2],[5,3.5,1.3,0.3],[4.5,2.3,1.3,0.3],[4.4,3.2,1.3,0.2],[5,3.5,1.6,0.6],[5.1,3.8,1.9,0.4],[4.8,3,1.4,0.3],[5.1,3.8,1.6,0.2],[4.6,3.2,1.4,0.2],[5.3,3.7,1.5,0.2],[5,3.3,1.4,0.2]],
        [[7,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],[5.5,2.3,4,1.3],[6.5,2.8,4.6,1.5],[5.7,2.8,4.5,1.3],[6.3,3.3,4.7,1.6],[4.9,2.4,3.3,1],[6.6,2.9,4.6,1.3],[5.2,2.7,3.9,1.4],[5,2,3.5,1],[5.9,3,4.2,1.5],[6,2.2,4,1],[6.1,2.9,4.7,1.4],[5.6,2.9,3.6,1.3],[6.7,3.1,4.4,1.4],[5.6,3,4.5,1.5],[5.8,2.7,4.1,1],[6.2,2.2,4.5,1.5],[5.6,2.5,3.9,1.1],[5.9,3.2,4.8,1.8],[6.1,2.8,4,1.3],[6.3,2.5,4.9,1.5],[6.1,2.8,4.7,1.2],[6.4,2.9,4.3,1.3],[6.6,3,4.4,1.4],[6.8,2.8,4.8,1.4],[6.7,3,5,1.7],[6,2.9,4.5,1.5],[5.7,2.6,3.5,1],[5.5,2.4,3.8,1.1],[5.5,2.4,3.7,1],[5.8,2.7,3.9,1.2],[6,2.7,5.1,1.6],[5.4,3,4.5,1.5],[6,3.4,4.5,1.6],[6.7,3.1,4.7,1.5],[6.3,2.3,4.4,1.3],[5.6,3,4.1,1.3],[5.5,2.5,4,1.3],[5.5,2.6,4.4,1.2],[6.1,3,4.6,1.4],[5.8,2.6,4,1.2],[5,2.3,3.3,1],[5.6,2.7,4.2,1.3],[5.7,3,4.2,1.2],[5.7,2.9,4.2,1.3],[6.2,2.9,4.3,1.3],[5.1,2.5,3,1.1],[5.7,2.8,4.1,1.3]],
        [[6.3,3.3,6,2.5],[5.8,2.7,5.1,1.9],[7.1,3,5.9,2.1],[6.3,2.9,5.6,1.8],[6.5,3,5.8,2.2],[7.6,3,6.6,2.1],[4.9,2.5,4.5,1.7],[7.3,2.9,6.3,1.8],[6.7,2.5,5.8,1.8],[7.2,3.6,6.1,2.5],[6.5,3.2,5.1,2],[6.4,2.7,5.3,1.9],[6.8,3,5.5,2.1],[5.7,2.5,5,2],[5.8,2.8,5.1,2.4],[6.4,3.2,5.3,2.3],[6.5,3,5.5,1.8],[7.7,3.8,6.7,2.2],[7.7,2.6,6.9,2.3],[6,2.2,5,1.5],[6.9,3.2,5.7,2.3],[5.6,2.8,4.9,2],[7.7,2.8,6.7,2],[6.3,2.7,4.9,1.8],[6.7,3.3,5.7,2.1],[7.2,3.2,6,1.8],[6.2,2.8,4.8,1.8],[6.1,3,4.9,1.8],[6.4,2.8,5.6,2.1],[7.2,3,5.8,1.6],[7.4,2.8,6.1,1.9],[7.9,3.8,6.4,2],[6.4,2.8,5.6,2.2],[6.3,2.8,5.1,1.5],[6.1,2.6,5.6,1.4],[7.7,3,6.1,2.3],[6.3,3.4,5.6,2.4],[6.4,3.1,5.5,1.8],[6,3,4.8,1.8],[6.9,3.1,5.4,2.1],[6.7,3.1,5.6,2.4],[6.9,3.1,5.1,2.3],[5.8,2.7,5.1,1.9],[6.8,3.2,5.9,2.3],[6.7,3.3,5.7,2.5],[6.7,3,5.2,2.3],[6.3,2.5,5,1.9],[6.5,3,5.2,2],[6.2,3.4,5.4,2.3],[5.9,3,5.1,1.8]]
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


from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
print('labels=',labels)
centroids = kmeans.cluster_centers_
print('centroids =',centroids)

X=X.tolist()

#db= calculate_davies_bouldin_index(data_points, data_point_labels)
db= calculate_davies_bouldin_index(X, labels)
print('db=',db)
#dn=calculate_dunn_index(data_points, data_point_labels)
dn=calculate_dunn_index(X, labels)
print('dn=',dn)
#data_points = np.array(data_points)
db_score = davies_bouldin_score(data_points, data_point_labels)
print('db_score=',db_score)
print('SI =',dn-db)