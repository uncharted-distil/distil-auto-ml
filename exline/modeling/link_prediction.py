#!/usr/bin/env python

"""
    exline/modeling/link_prediction.py
    
    
"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.model_selection import KFold

from rescal import rescal_als

from .metrics import metrics
from ..utils import parmap

def edgelist2tensor(edgelist, num_nodes, num_edge_types):
    adj = [sparse.lil_matrix((num_nodes, num_nodes)) for _ in range(num_edge_types)]
    for ii, jj, kk in edgelist:
        adj[kk][ii, jj] = 1
    
    return adj


def nx2edgelist(graph, attr_name='linkType'):
    i, j, _  = list(zip(*graph.edges))
    k        = list(nx.get_edge_attributes(graph, attr_name).values())
    edgelist = np.column_stack([i, j, k])
    return edgelist.astype(np.int64)


def rescal_link_prediction(adj, rank=100, lambda_A=10, lambda_R=10, conv=1e-3, maxIter=500):
    num_nodes      = adj[0].shape[0]
    num_edge_types = len(adj)
    
    A, R, _, _, _ = rescal_als(
        X=adj,
        rank=rank,
        lambda_A=lambda_A,
        lambda_R=lambda_R,
        conv=conv,
        maxIter=maxIter,
    )
    
    adj_lr = np.stack([(A @ R[k] @ A.T) for k in range(num_edge_types)], axis=-1)
    
    adj_lr /= (np.linalg.norm(adj_lr, axis=-1, keepdims=True) + 1e-10) # Normalize by link type
    
    return adj_lr


def sample_missing_edges(edgelist, n):
    """ randomly sample zero entries from edgelist """
    zero_entries = np.column_stack([np.random.choice(max(edgelist[:,i]), 2 * n) for i in range(3)])
    zero_entries = np.vstack(set([tuple(z) for z in zero_entries]).difference([tuple(nz) for nz in edgelist]))
    zero_sel     = np.random.choice(zero_entries.shape[0], n, replace=False)
    zero_entries = zero_entries[zero_sel]
    return zero_entries


def cv_fold(edgelist, num_nodes, num_edge_types, train_idx, valid_idx, target_metric):
    
    n_edges = edgelist.shape[0]
    n_train = train_idx.shape[0]
    n_valid = valid_idx.shape[0]
    
    zero_entries   = sample_missing_edges(edgelist, n_valid)
    edgelist_train = edgelist[train_idx]
    
    edgelist_valid   = np.row_stack([edgelist[valid_idx], zero_entries])
    y_valid          = np.hstack([np.ones(n_valid), np.zeros(n_valid)])
    
    adj_train    = edgelist2tensor(edgelist_train, num_nodes, num_edge_types)
    adj_train_lr = rescal_link_prediction(adj_train)
    
    scores_valid = adj_train_lr[tuple(edgelist_valid.T)]
    
    threshs = np.linspace(-1, 1, 512)
    scores  = [metrics[target_metric](y_valid, scores_valid > i) for i in threshs]
    
    return {
        "thresh" : threshs[np.argmax(scores)],
        "score"  : scores[np.argmax(scores)],
    }


class RescalLinkPrediction:
    
    def __init__(self, target_metric):
        
        self.target_metric = target_metric
        self.adj_lr = None
    
    def fit(self, graph, X_train, y_train):
        global _cv_fold
        
        edgelist = nx2edgelist(graph)
        
        num_nodes      = len(graph.nodes())
        num_edge_types = len(set(edgelist[:,-1]))
        
        # --
        # Use CV to estimate optimal threshold
        
        def _cv_fold(args):
            train_idx, valid_idx = args
            return cv_fold(edgelist, num_nodes, num_edge_types, train_idx, valid_idx, target_metric=self.target_metric)
            
        cv_res = parmap(_cv_fold, KFold(n_splits=10, shuffle=True).split(edgelist))
        cv_res = pd.DataFrame(cv_res)
        self.opt_thresh = cv_res.thresh.mean()
        
        # --
        # Factorize whole matrix
        
        adj         = edgelist2tensor(edgelist, num_nodes, num_edge_types)
        self.adj_lr = rescal_link_prediction(adj)
        
        return self
    
    def predict(self, X):
        return self.adj_lr[(
            X.source_nodeID.values,
            X.target_nodeID.values,
            X.linkType.values,
        )] > self.opt_thresh
