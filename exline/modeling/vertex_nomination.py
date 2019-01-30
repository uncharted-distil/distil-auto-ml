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
    
    def fit_score(self, graph, X_train, X_test, y_train, y_test):
        assert X_train.shape[1] == 1
        assert X_test.shape[1] == 1
        
        X_train.columns = ('nodeID',)
        X_test.columns  = ('nodeID',)
        
        # --
        # Featurize
        
        df = pd.DataFrame([graph.nodes[i] for i in graph.nodes]).set_index('nodeID')
        
        adj = nx.adjacency_matrix(graph).astype(np.float64)
        U, _, _ = linalg.svds(adj, k=self.num_components)
        
        Xf_train = np.hstack([df.loc[X_train.nodeID].values, U[X_train.nodeID.values]])
        Xf_test  = np.hstack([df.loc[X_test.nodeID].values, U[X_test.nodeID.values]])
        
        # --
        # Choose the best model
        
        print('VertexNominationCV: ForestCV', file=sys.stderr)
        forest = ForestCV(target_metric=self.target_metric)
        forest = forest.fit(Xf_train, y_train)
        
        print('VertexNominationCV: SupportVectorCV', file=sys.stderr)
        svm = SupportVectorCV(target_metric=self.target_metric)
        svm = svm.fit(Xf_train, y_train)
        
        if (svm.best_fitness > forest.best_fitness):
            self.test_pred   = svm.model.predict(Xf_test)
            self.best_params = svm.best_params
            self.score_cv    = svm.best_fitness
        else:
            self.test_pred    = forest.predict(Xf_test)
            self.best_params  = forest.best_params
            self.best_fitness = forest.best_fitness
        
        return metrics[self.target_metric](y_test, self.test_pred)