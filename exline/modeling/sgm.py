#!/usr/bin/env python

"""
    exline/modeling/sgm.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse

from .base import EXLineBaseModel
from .metrics import metrics
from sgm.backends.classic import ScipyJVClassicSGM

# --
# Helpers

def make_preds(P, X):
    preds = P[(X.num_id1.values, X.num_id2.values)]
    preds = np.asarray(preds).squeeze()
    return preds


def pad_graphs(G1, G2):
    n_nodes = max(G1.order(), G2.order())
    
    for i in range(n_nodes - G1.order()):
        G1.add_node('__new_node__salt123_%d' % i)
        
    for i in range(n_nodes - G2.order()):
        G2.add_node('__new_node__salt456_%d' % i)
        
    assert G1.order() == G2.order()
    
    return G1, G2, n_nodes


class SGMGraphMatcher(EXLineBaseModel):
    def __init__(self, target_metric, num_iters=20, tolerance=1, 
        verbose=True, unweighted=True):
        
        assert target_metric == 'accuracy'
        
        self.target_metric = target_metric
        
        self.num_iters = num_iters
        self.tolerance = tolerance
        self.verbose   = verbose
        self.unweighted = unweighted
    
    def predict(self, X):
        assert X.shape[1] == 2
        X.columns  = ('orig_id1', 'orig_id2')
        
        X.orig_id1  = X.orig_id1.astype(str)
        X.orig_id2  = X.orig_id2.astype(str)
        
        X['num_id1']  = X['orig_id1'].apply(lambda x: self.G1_lookup[x])
        X['num_id2']  = X['orig_id2'].apply(lambda x: self.G2_lookup[x])
        
        return make_preds(self.P, X)
    
    def fit(self, X_train, y_train, U_train):
        graphs = U_train['graphs']
        
        assert list(graphs.keys()) == ['0', '1']
        assert X_train.shape[1] == 2
        
        G1 = graphs['0']
        G2 = graphs['1'] # !! assumes this is correct
        
        assert isinstance(list(G1.nodes)[0], str)
        assert isinstance(list(G2.nodes)[0], str)
        
        X_train.columns  = ('orig_id1', 'orig_id2')
        X_train.orig_id1 = X_train.orig_id1.astype(str)
        X_train.orig_id2 = X_train.orig_id2.astype(str)
        
        G1, G2, n_nodes = pad_graphs(G1, G2)
        
        self.G1_lookup = dict(zip(G1.nodes, range(len(G1.nodes))))
        self.G2_lookup = dict(zip(G2.nodes, range(len(G2.nodes))))
        
        X_train['num_id1'] = X_train['orig_id1'].apply(lambda x: self.G1_lookup[x])
        X_train['num_id2'] = X_train['orig_id2'].apply(lambda x: self.G2_lookup[x])
        
        # --
        # Convert to matrix
        
        G1p = nx.relabel_nodes(G1, self.G1_lookup)
        G2p = nx.relabel_nodes(G2, self.G2_lookup)
        A   = nx.adjacency_matrix(G1p, nodelist=list(self.G1_lookup.values()))
        B   = nx.adjacency_matrix(G2p, nodelist=list(self.G2_lookup.values()))
        
        if self.unweighted:
            A = (A != 0)
            B = (B != 0)
            
        # Symmetrize (required by our SGM implementation)
        # Does it hurt performance?
        A = ((A + A.T) > 0).astype(np.float32)
        B = ((B + B.T) > 0).astype(np.float32)
        
        P = X_train[['num_id1', 'num_id2']][y_train == 1].values
        P = sparse.csr_matrix((np.ones(P.shape[0]), (P[:,0], P[:,1])), shape=(n_nodes, n_nodes))
        
        sgm = ScipyJVClassicSGM(A=A, B=B, P=P, verbose=self.verbose)
        P_out = sgm.run(
            num_iters=self.num_iters,
            tolerance=self.tolerance
        )
        P_out = sparse.csr_matrix((np.ones(n_nodes), (np.arange(n_nodes), P_out)))
        
        P_eye = sparse.eye(P.shape[0]).tocsr()
        
        self.sgm_train_acc  = metrics[self.target_metric](y_train, make_preds(P_out, X_train))
        self.null_train_acc = metrics[self.target_metric](y_train, make_preds(P_eye, X_train))
        
        if self.null_train_acc > self.sgm_train_acc:
            self.P = P_eye
        else:
            self.P = P_out
        
        return self
    
    @property
    def details(self):
        return {
            "train_acc"      : self.sgm_train_acc,
            "null_train_acc" : self.null_train_acc,
        }
