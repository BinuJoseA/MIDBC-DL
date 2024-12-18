# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:57:44 2024

@author: BinuJoseA
"""


import numpy as np
import pandas as pd
from cluster import cluster


class incdbscanner:
    
    def __init__(self):
        self.dataSet = []
        self.count = 0
        self.curCores = []
        self.Clusters = []
        self.Noise = cluster('Noise')
        self.num = 0

    def incdbscan(self, D, eps, MinPts):
        for point in D:
            self.dataSet.append(point)
            self.incrementalAdd(point, eps, MinPts)
            
        self.cleanupClusters()
        self.printClusters()
        
        
    
    def expandCluster(self, point, NeighbourPoints, C, eps, MinPts):
        C.addPoint(point)
        for p in NeighbourPoints:
            if not any(np.array_equal(p, visited_point) for visited_point in self.visited):
                self.visited.append(p)
                n2 = self.regionQuery(p, eps)
                if len(n2) >= MinPts:
                    for n in n2:
                        if not any(np.array_equal(n, neighbour_point) for neighbour_point in NeighbourPoints):
                            NeighbourPoints.append(n)
                    
            for c in self.Clusters:
                if not c.has(p):
                    if not C.has(p):
                        C.addPoint(p)
                        
            if not self.Clusters:
                if not C.has(p):
                    C.addPoint(p)
                        
        self.Clusters.append(C)
    

    def mergeClusters(self, mergthld):
        """
        Merges clusters in self.Clusters if the distance between them is below the merging threshold.
        """
        merged_clusters = []
        skip_clusters = set()
    
        for i, cluster_a in enumerate(self.Clusters):
            if i in skip_clusters:
                continue
    
            for j, cluster_b in enumerate(self.Clusters):
                if j <= i or j in skip_clusters:
                    continue
    
                # Check if clusters should be merged
                if self.shouldMerge(cluster_a, cluster_b, mergthld):
                    cluster_a.merge(cluster_b)  # Merge cluster_b into cluster_a
                    skip_clusters.add(j)  # Mark cluster_b as merged
    
            merged_clusters.append(cluster_a)
    
        self.Clusters = merged_clusters

    def shouldMerge(self, cluster_a, cluster_b, mergthld):
        """
        Determines whether two clusters should be merged based on the merging threshold.
        """
        for point_a in cluster_a.points:
            for point_b in cluster_b.points:
                if np.linalg.norm(point_a - point_b) <= mergthld:
                    return True
        return False

    def regionQuery(self, P, eps):
        result = []
        for d in self.dataSet:
            if np.sqrt(np.sum((np.array(d) - np.array(P)) ** 2)) <= eps:
                result.append(d)
        return result
    '''
    def regionQuery(self, P, eps):
        result = []
        for d in self.dataSet:
            # Convert each element of the list to float and perform the calculation
            if np.sqrt(np.sum((np.array(d, dtype=float) - np.array(P, dtype=float)) ** 2)) <= eps:
                result.append(d)
        return result
    '''
    def incrementalAdd(self, p, eps, MinPts):
        self.num = self.num + 1
        #print('------------------------------------')
        #print("\nADDING point " +  ' --> ', p)
        self.visited = []
        self.newCores = []
        UpdSeedIns = []
        NeighbourPoints = self.regionQuery(p, eps)
        if(len(NeighbourPoints) >= MinPts):
            self.newCores.append(p)
        self.visited.append(p)
        
        for clust in self.Clusters:
            if clust.has(p):
                clust.remPoint(p)
        
        for pt in NeighbourPoints:
            if not any(np.array_equal(pt, visited_pt) for visited_pt in self.visited):
                np5 = self.regionQuery(pt, eps)
                if len(np5) >= MinPts:
                    for n in np5:
                        if not any(np.array_equal(n, npx) for npx in NeighbourPoints):
                            NeighbourPoints.append(n)
                    if not any(np.array_equal(pt, core1) for core1 in self.curCores):
                        self.newCores.append(pt)
        for core in self.newCores:
            corehood = self.regionQuery(core, eps)
            for elem in corehood:
                if len(self.regionQuery(elem, eps)) >= MinPts:
                    if not any(np.array_equal(elem, seed1) for seed1 in UpdSeedIns):
                        UpdSeedIns.append(elem)
        new_cluster = None
        if len(UpdSeedIns) < 1:
            self.Noise.addPoint(p)
        else:
            findCount = 0
            foundClusters = []
            for seed in UpdSeedIns:
                for clust in self.Clusters:
                    if clust.has(seed):
                        findCount += 1
                        if clust.name not in foundClusters:
                            foundClusters.append(clust.name)
                            break
            if len(foundClusters) == 0:
                name = 'Cluster' + str(self.count)
                new_cluster = cluster(name)
                self.count += 1
                self.expandCluster(UpdSeedIns[0], self.regionQuery(UpdSeedIns[0], eps), new_cluster, eps, MinPts)
            elif len(foundClusters) == 1:
                originalCluster = None
                newCluster = None
                for c in self.Clusters:
                    if c.name == foundClusters[0]:
                        originalCluster = c
                        newCluster = c
                newCluster.addPoint(p)
                if len(UpdSeedIns) > findCount:
                    for seed in UpdSeedIns:
                        if not newCluster.has(seed):
                            newCluster.addPoint(seed)
                self.Clusters.remove(originalCluster)
                self.Clusters.append(newCluster)
            else:
                masterCluster = None
                originalCluster = None
                for c in self.Clusters:
                    if c.name == foundClusters[0]:
                        masterCluster = c
                    if c.name == foundClusters[1]:
                        originalCluster = c
    
                #is_contained = all(point in masterCluster.getPoints() for point in originalCluster.getPoints())
                is_contained = any(np.array_equal(point, master_point) for point in originalCluster.getPoints() for master_point in masterCluster.getPoints())
    
                if is_contained:
                    for clusname in foundClusters:
                        for clus in self.Clusters:
                            if clus.name == clusname:
                                for cluspoints in clus.getPoints():
                                    if not masterCluster.has(cluspoints):
                                        masterCluster.addPoint(cluspoints)
                    if len(UpdSeedIns) > findCount:
                        for seed in UpdSeedIns:
                            if not masterCluster.has(seed):
                                masterCluster.addPoint(seed)
                    self.Clusters.remove(originalCluster)
                else:
                    for clusname in foundClusters:
                        for clus in self.Clusters:
                            if clus.name == clusname:
                                for cluspoints in clus.getPoints():
                                    if not masterCluster.has(cluspoints):
                                        masterCluster.addPoint(cluspoints)
                    if len(UpdSeedIns) > findCount:
                        for seed in UpdSeedIns:
                            if not masterCluster.has(seed):
                                masterCluster.addPoint(seed)
        
        # Remove individual clusters that are entirely contained within the merged cluster
        individual_clusters_to_remove = []
        if new_cluster is not None:
            for cluster1 in self.Clusters:
                if cluster1.name != new_cluster.name:
                    #is_contained = all(point in new_cluster.getPoints() for point in cluster1.getPoints())
                    is_contained = any(np.array_equal(point, new_point) for point in cluster1.getPoints() for new_point in new_cluster.getPoints())
                    if is_contained:
                        individual_clusters_to_remove.append(cluster1)
    
        for cluster2 in individual_clusters_to_remove:
            self.Clusters.remove(cluster2)
    
    def cleanupClusters(self):
        Noisepoints = []
        for noisep in self.Noise.getPoints():
            Noisepoints.append(noisep)
        for pts in Noisepoints:
            for clust in self.Clusters:
                if clust.has(pts):
                    self.Noise.remPoint(pts)
    
        #print('Noise =', self.Noise.getPoints())
    
        # Find individual clusters to remove
        individual_clusters_to_remove = []
        for cluster1 in self.Clusters:
            for cluster2 in self.Clusters:
                if cluster1.name != cluster2.name:
                    is_contained = all(point in cluster2.getPoints() for point in cluster1.getPoints())
                    if is_contained:
                        individual_clusters_to_remove.append(cluster1)
    
        # Remove individual clusters
        for cluster_to_remove in individual_clusters_to_remove:
            if cluster_to_remove in self.Clusters:
                self.Clusters.remove(cluster_to_remove)


        
        
    def printClusters(self):
        print('Cluster list =')
        #print(self.Clusters)
        for clust in self.Clusters:
            
            print(clust.getPoints())
        #print('inside printClusters')
        #print(self.Clusters)
