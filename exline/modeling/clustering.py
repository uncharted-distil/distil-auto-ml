#!/usr/bin/env python

"""
    exline/modeling/clustering.py
"""

from .metrics import metrics

from sklearn.cluster import KMeans

class ClusteringCV:
    
    def __init__(self, target_metric, n_clusters, all_float):
        
        self.target_metric = target_metric
        self.n_clusters    = n_clusters
        self.all_float     = all_float
        self.n_init        = 100
    
    def fit(self, X_train, y_train):
        assert X_train.shape[0] == 0
        assert self.all_float
        
        print('!! ClusteringCV.fit does nothing')
        return self
    
    def predict(self, X):
        self.model = KMeans(n_clusters=self.n_clusters, n_init=self.n_init)
        return self.model.fit_predict(X)

