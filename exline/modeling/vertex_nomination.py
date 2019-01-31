#!/usr/bin/env python

"""
    exline/modeling/vertex_nomination.py
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import linalg

from .forest import ForestCV
from .svm import SupportVectorCV
from .metrics import metrics, classification_metrics

class VertexNominationCV:
    
    def __init__(self, target_metric, num_components=8):
        self.target_metric  = target_metric
        self.num_components = num_components
        
        self.feats = None
    
    def fit(self, X_train, y_train, U_train=None):
        graph = U_train['graph']
        X_train = X_train.copy()
        assert X_train.shape[1] == 1
        
        X_train.columns = ('nodeID',)
        
        # --
        # Featurize
        
        df = pd.DataFrame([graph.nodes[i] for i in graph.nodes]).set_index('nodeID')
        
        adj = nx.adjacency_matrix(graph).astype(np.float64)
        U, _, _ = linalg.svds(adj, k=self.num_components)
        
        self.feats = pd.DataFrame(np.hstack([df.values, U])).set_index(df.index)
        
        Xf_train = self.feats.loc[X_train.nodeID].values
        
        # --
        # Choose the best model
        
        print('VertexNominationCV: ForestCV', file=sys.stderr)
        forest = ForestCV(target_metric=self.target_metric)
        forest = forest.fit(Xf_train, y_train)
        
        print('VertexNominationCV: SupportVectorCV', file=sys.stderr)
        svm = SupportVectorCV(target_metric=self.target_metric)
        svm = svm.fit(Xf_train, y_train)
        
        if (svm.best_fitness > forest.best_fitness):
            self.model       = svm.model
            self.best_params = svm.best_params
            self.score_cv    = svm.best_fitness
        else:
            self.model        = forest
            self.best_params  = forest.best_params
            self.best_fitness = forest.best_fitness
        
        return self
    
    def predict(self, X):
        X = X.copy()
        assert X.shape[1] == 1
        X.columns = ('nodeID',)
        
        Xf = self.feats.loc[X.nodeID].values
        return self.model.predict(Xf)