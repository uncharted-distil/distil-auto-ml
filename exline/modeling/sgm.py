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
from sklearn import metrics

from sgm.backends.classic import ScipyJVClassicSGM

# --
# Helpers

def compute_acc(P, X_train, X_test, y_train, y_test):
    null_train_preds = P[(X_train.num_id1.values, X_train.num_id2.values)]
    null_train_preds = np.asarray(null_train_preds).squeeze()
    null_train_acc   = metrics.accuracy_score(y_train, null_train_preds)
    
    null_test_preds = P[(X_test.num_id1.values, X_test.num_id2.values)]
    null_test_preds = np.asarray(null_test_preds).squeeze()
    null_test_acc   = metrics.accuracy_score(y_test, null_test_preds)
    
    return null_train_acc, null_test_acc

def pad_graphs(G1, G2):
    n_nodes = max(G1.order(), G2.order())
    
    for i in range(n_nodes - G1.order()):
        G1.add_node('__new_node__salt123_%d' % i)
        
    for i in range(n_nodes - G2.order()):
        G2.add_node('__new_node__salt456_%d' % i)
        
    assert G1.order() == G2.order()
    
    return G1, G2, n_nodes


class SGMGraphMatcher:
    def __init__(self, target_metric, num_iters=20, tolerance=1, verbose=True):
        assert target_metric == 'accuracy'
        
        self.target_metric = target_metric
        
        self.num_iters = num_iters
        self.tolerance = tolerance
        self.verbose   = verbose


    def fit(self, graphs, X_train, X_test, y_train, y_test, unweighted=True):
        assert list(graphs.keys()) == ['0', '1']
        assert X_train.shape[1] == 2
        assert X_test.shape[1] == 2
        
        G1 = graphs['0']
        G2 = graphs['1'] # !! assumes this is correct
        
        assert isinstance(list(G1.nodes)[0], str)
        assert isinstance(list(G2.nodes)[0], str)
        
        X_train.columns = ('orig_id1', 'orig_id2')
        #X_test.columns  = ('orig_id1', 'orig_id2')
        
        X_train.orig_id1 = X_train.orig_id1.astype(str)
        X_train.orig_id2 = X_train.orig_id2.astype(str)
       # X_test.orig_id1  = X_test.orig_id1.astype(str)
        #X_test.orig_id2  = X_test.orig_id2.astype(str)
        
        G1, G2, n_nodes = pad_graphs(G1, G2)
        
        G1_nodes = sorted(dict(G1.degree()).items(), key=lambda x: -x[1])
        G1_nodes = list(zip(*G1_nodes))[0]
        G1_lookup = dict(zip(G1.nodes, range(len(G1.nodes))))
        X_train['num_id1'] = X_train['orig_id1'].apply(lambda x: G1_lookup[x])
        #X_test['num_id1']  = X_test['orig_id1'].apply(lambda x: G1_lookup[x])
                
        G2_nodes = sorted(dict(G1.degree()).items(), key=lambda x: -x[1])
        G2_nodes = list(zip(*G2_nodes))[0]
        G2_lookup = dict(zip(G2.nodes, range(len(G2.nodes))))
        X_train['num_id2'] = X_train['orig_id2'].apply(lambda x: G2_lookup[x])
        #X_test['num_id2']  = X_test['orig_id2'].apply(lambda x: G2_lookup[x])
        
        # --
        # Convert to matrix
        G1p = nx.relabel_nodes(G1, G1_lookup)
        G2p = nx.relabel_nodes(G2, G2_lookup)
        A   = nx.adjacency_matrix(G1p, nodelist=list(G1_lookup.values()))
        B   = nx.adjacency_matrix(G2p, nodelist=list(G2_lookup.values()))
        
        if unweighted:
            A = (A != 0)
            B = (B != 0)
            
        # Symmetrize (required by our SGM implementation)
        # Does it hurt performance?
        A = ((A + A.T) > 0).astype(np.float32)
        B = ((B + B.T) > 0).astype(np.float32)
        
        P = X_train[['num_id1', 'num_id2']].values
        P = sparse.csr_matrix((np.ones(P.shape[0]), (P[:,0], P[:,1])), shape=(n_nodes, n_nodes))
        
        sgm = ScipyJVClassicSGM(A=A, B=B, P=P, verbose=self.verbose)
        P_out = sgm.run(
            num_iters=self.num_iters,
            tolerance=self.tolerance
        )
        return P_out

    
    def fit_score(self, graphs, X_train, X_test, y_train, y_test, unweighted=True):
        assert list(graphs.keys()) == ['0', '1']
        assert X_train.shape[1] == 2
        assert X_test.shape[1] == 2
        
        G1 = graphs['0']
        G2 = graphs['1'] # !! assumes this is correct
        
        assert isinstance(list(G1.nodes)[0], str)
        assert isinstance(list(G2.nodes)[0], str)
        
        X_train.columns = ('orig_id1', 'orig_id2')
        X_test.columns  = ('orig_id1', 'orig_id2')
        
        X_train.orig_id1 = X_train.orig_id1.astype(str)
        X_train.orig_id2 = X_train.orig_id2.astype(str)
        X_test.orig_id1  = X_test.orig_id1.astype(str)
        X_test.orig_id2  = X_test.orig_id2.astype(str)
        
        G1, G2, n_nodes = pad_graphs(G1, G2)
        
        G1_nodes = sorted(dict(G1.degree()).items(), key=lambda x: -x[1])
        G1_nodes = list(zip(*G1_nodes))[0]
        G1_lookup = dict(zip(G1.nodes, range(len(G1.nodes))))
        X_train['num_id1'] = X_train['orig_id1'].apply(lambda x: G1_lookup[x])
        X_test['num_id1']  = X_test['orig_id1'].apply(lambda x: G1_lookup[x])
                
        G2_nodes = sorted(dict(G1.degree()).items(), key=lambda x: -x[1])
        G2_nodes = list(zip(*G2_nodes))[0]
        G2_lookup = dict(zip(G2.nodes, range(len(G2.nodes))))
        X_train['num_id2'] = X_train['orig_id2'].apply(lambda x: G2_lookup[x])
        X_test['num_id2']  = X_test['orig_id2'].apply(lambda x: G2_lookup[x])
        
        # --
        # Convert to matrix
        
        G1p = nx.relabel_nodes(G1, G1_lookup)
        G2p = nx.relabel_nodes(G2, G2_lookup)
        A   = nx.adjacency_matrix(G1p, nodelist=list(G1_lookup.values()))
        B   = nx.adjacency_matrix(G2p, nodelist=list(G2_lookup.values()))
        
        if unweighted:
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
        
        train_acc, test_acc = compute_acc(P_out, X_train, X_test, y_train, y_test)
        
        # Compare our results to null model.  If worse on train data, use null model.
        P_null = sparse.eye(P.shape[0]).tocsr()
        null_train_acc, null_test_acc = compute_acc(P_null, X_train, X_test, y_train, y_test)
        best_acc = null_test_acc if null_train_acc > train_acc else test_acc
        
        self.train_acc      = train_acc
        self.test_acc       = test_acc
        self.null_train_acc = null_train_acc
        self.null_test_acc  = null_test_acc
        
        return best_acc