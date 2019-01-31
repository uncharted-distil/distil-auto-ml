#!/usr/bin/env python

"""
    exline/modeling/clustering.py
"""

from sklearn.cluster import KMeans

from .base import EXLineBaseModel
from .metrics import metrics

class ClusteringCV(EXLineBaseModel):
    
    def __init__(self, target_metric, n_clusters, all_float):
        
        self.target_metric = target_metric
        self.n_clusters    = n_clusters
        self.all_float     = all_float
        self.n_init        = 100
    
    def fit(self, X_train, y_train, U_train=None):
        assert X_train.shape[0] == 0
        assert self.all_float
        
        print('!! ClusteringCV.fit does nothing')
        return self
    
    def predict(self, X):
        self.model = KMeans(n_clusters=self.n_clusters, n_init=self.n_init)
        return self.model.fit_predict(X)

