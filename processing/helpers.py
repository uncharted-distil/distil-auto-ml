#!/usr/bin/env python

"""
    helpers.py
"""

import functools
import numpy as np
import pandas as pd
import multiprocessing
from time import time

from scipy import sparse
from joblib import Parallel, delayed

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def parmap(fn, x, n_jobs=32, backend='multiprocessing', verbose=1, **kwargs):
    jobs = [delayed(fn)(xx, **kwargs) for xx in x]
    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(jobs)


def with_timeout(timeout):
    def decorator(decorated):
        @functools.wraps(decorated)
        def inner(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(1)
            async_result = pool.apply_async(decorated, args, kwargs)
            try:
                return async_result.get(timeout)
            except multiprocessing.TimeoutError:
                return
            
        return inner
    return decorator


def remap_labels(y_train, y_test):
    ulab     = list(set(y_train))
    y_lookup = dict(zip(ulab, range(len(ulab))))
    y_train  = np.array([y_lookup.get(yy) for yy in y_train])
    y_test   = np.array([y_lookup.get(yy) for yy in y_test])
    return y_train, y_test, y_lookup


def compute_win(df, our_score='beeline_score', their_score='ll_score', metric='metric'):
    return (
        ((df[our_score] >= df[their_score]) & (df[metric] == 'f1Macro')) | 
        ((df[our_score] <= df[their_score]) & (df[metric] == 'meanSquaredError'))
    )


def dense2sparse(X):
    vals = np.hstack(X)
    rows = np.repeat(np.arange(X.shape[0]), X.shape[1])
    cols = np.tile(np.arange(X.shape[1]), X.shape[0])
    return sparse.csr_matrix((vals, (rows, cols)))


# def benchmark_rf(Xf_train, y_train, metric):
#     if metric == 'f1Macro':
#         model_cls = RandomForestClassifier
#     elif metric == 'meanSquaredError':
#         model_cls = RandomForestRegressor
#     else:
#         raise Exception
    
#     t = time()
#     _ = model_cls(n_estimators=512).fit(Xf_train, y_train)
#     return time() - t

d3m_metrics = {
    'f1Macro'                     : lambda act, pred: metrics.f1_score(act, pred, average='macro'),
    'f1'                          : lambda act, pred: metrics.f1_score(act, pred),
    'meanSquaredError'            : metrics.mean_squared_error,
    'meanAbsoluteError'           : metrics.mean_absolute_error,
    'rootMeanSquaredError'        : lambda act, pred: np.sqrt(metrics.mean_squared_error(act, pred)),
    'rootMeanSquaredErrorAvg'     : lambda act, pred: np.sqrt(metrics.mean_squared_error(act, pred)),
    'accuracy'                    : metrics.accuracy_score,
    'normalizedMutualInformation' : metrics.normalized_mutual_info_score
}

# --
# Ensemble voting

def _vote(p, tiebreaker):
    cnts = np.bincount(p)
    if (cnts == cnts.max()).sum() > 1:
        # if tie, break according to tiebreaker
        top = np.where(cnts == cnts.max())[0]
        return top[np.argmin([tiebreaker.index(t) for t in top])]
    else:
        return cnts.argmax()

def vote(preds, y_train):
    # Vote, breaking ties according to class prevalance
    tiebreaker = list(pd.value_counts(y_train).index)
    return np.array([_vote(p, tiebreaker) for p in preds.T])

# def argmax_prevalance(x):
# Argmax, breaking ties according to class prevalance