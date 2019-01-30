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
    
    def fit_score(self, X_train, X_test, y_train, y_test):
        assert X_train.shape[0] == 0
        assert self.all_float
        
        self.model = KMeans(n_clusters=self.n_clusters, n_init=100)
        pred_test  = self.model.fit_predict(X_test)
        
        return metrics[self.target_metric](y_test, pred_test)