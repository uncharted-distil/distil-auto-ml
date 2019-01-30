#!/usr/bin/env python

"""
    exline/modeling/helpers.py
"""


import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import metrics

# ---------------------------------------------------------
# F1 macro adjusters

def _adjusted_f1_macro(probs, y, bias):
    return metrics['f1Macro'](y, (probs + bias).argmax(axis=-1))

def _adjust_f1_inner(probs_oob, y_train, alpha):
    def _make_adj(bias):
        score = _adjusted_f1_macro(probs_oob, y_train, bias)
        norm  = alpha * np.abs(bias).mean()
        return (- score) + norm
    
    bias = np.zeros((1, probs_oob.shape[1]))
    bias = minimize(_make_adj, bias, method='Powell').x
    return bias

def adjust_f1_macro(model, Xf_test, y_train, y_test, alpha=0.01, subset=-1):
    # !! y_train and y_test must be sequential numeric
    
    probs_oob = model.oob_decision_function_
    
    if (subset > 0) and (subset < probs_oob.shape[0]):
        sel = np.sort(np.random.choice(probs_oob.shape[0], subset, replace=False))
        bias = _adjust_f1_inner(probs_oob[sel], y_train[sel], alpha=alpha)
    else:
        bias = _adjust_f1_inner(probs_oob, y_train, alpha=alpha)
    
    probs_test = model.predict_proba(Xf_test)
    
    print('_make_adj:train_null', _adjusted_f1_macro(probs_oob, y_train, 0))
    print('_make_adj:train_bias', _adjusted_f1_macro(probs_oob, y_train, bias))
    print('_make_adj:test_null', _adjusted_f1_macro(probs_test, y_test, 0))
    print('_make_adj:test_bias', _adjusted_f1_macro(probs_test, y_test, bias))
    print('--')
    return _adjusted_f1_macro(probs_test, y_test, bias)

# --
# Ensembling

def _vote(p, tiebreaker):
    cnts = np.bincount(p)
    if (cnts == cnts.max()).sum() > 1:
        # if tie, break according to tiebreaker
        top = np.where(cnts == cnts.max())[0]
        return top[np.argmin([tiebreaker.index(t) for t in top])]
    else:
        return cnts.argmax()

def tiebreaking_vote(preds, y_train):
    # Vote, breaking ties according to class prevalance
    tiebreaker = list(pd.value_counts(y_train).index)
    return np.array([_vote(p, tiebreaker) for p in preds.T])