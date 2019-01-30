#!/usr/bin/env python

"""
    exline/preprocessing/transformers.py
"""

import sys
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.cluster import KMeans, MiniBatchKMeans

from .utils import MISSING_VALUE_INDICATOR, SINGLETON_INDICATOR

# --
# Categorical

class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.lookup = None
    
    def fit(self, X):
        levels = list(set(X.squeeze())) + [MISSING_VALUE_INDICATOR] # !! How to sort? Randomly? Alphabetically?
        levels = np.random.permutation(levels)
        
        vals      = range(len(levels))
        max_width = len(np.binary_repr(max(vals)))
        vals      = [np.binary_repr(v, width=max_width) for v in vals]
        vals      = [np.array(list(v)).astype(int) for v in vals]
        
        self.lookup = dict(zip(levels, vals))
        return self
    
    def transform(self, X, y=None):
        assert self.lookup is not None
        return np.vstack([self.lookup.get(xx, self.lookup[MISSING_VALUE_INDICATOR]) for xx in X.squeeze()])
    
    def fit_transform(self, X, y=None, **kwargs):
        _ = self.fit(X)
        return self.transform(X)


# --
# Text

class SVMTextEncoder(BaseEstimator, TransformerMixin):
    # !! add tuning
    def __init__(self):
        super().__init__()
        
        self._vect  = TfidfVectorizer(ngram_range=[1, 2], max_features=30000)
        self._model = LinearSVC(class_weight='balanced')
    
    def fit(self, X, y):
        raise NotImplemented
    
    def transform(self, X):
        X = pd.Series(X.squeeze()).fillna(MISSING_VALUE_INDICATOR).values
        
        Xv  = self._vect.transform(X)
        out = self._model.decision_function(Xv)
        
        if len(out.shape) == 1:
            out = out.reshape(-1, 1)
        
        return out
    
    def fit_transform(self, X, y=None, **kwargs):
        assert y is not None, 'SVMTextEncoder.fit_transform requires y'
        
        X = pd.Series(X.squeeze()).fillna(MISSING_VALUE_INDICATOR).values
        
        Xv  = self._vect.fit_transform(X)
        out = cross_val_predict(self._model, Xv, y, method='decision_function', n_jobs=3, cv=3)
        self._model = self._model.fit(Xv, y)
        
        if len(out.shape) == 1:
            out = out.reshape(-1, 1)
        
        return out

# --
# Dates

def _detect_date(X, n_sample=1000):
    try:
        # raise Exception # !! Why was this here?
        _ = pd.to_datetime(X.sample(n_sample, replace=True)) # Could also just look at schema
        return True
    except:
        return False


def enrich_dates(X_train, X_test):
    cols = list(X_train.columns)
    for c in cols:
        if (X_train[c].dtype == np.object_) and _detect_date(X_train[c]):
            print('-- detected date: %s' % c, file=sys.stderr)
            
            # try:
            train_seconds = (pd.to_datetime(X_train[c]) - pd.to_datetime('2000-01-01')).dt.total_seconds().values
            test_seconds = (pd.to_datetime(X_test[c]) - pd.to_datetime('2000-01-01')).dt.total_seconds().values
            
            sec_mean = train_seconds.mean()
            sec_std  = train_seconds.std()
            
            X_train['%s__seconds' % c] = (train_seconds - sec_mean) / sec_std
            X_test['%s__seconds' % c]  = (test_seconds - sec_mean) / sec_std
            # except:
                # X_train['%s__seconds' % c] = 0 
                # X_test['%s__seconds' % c]  = 0
    
    return X_train, X_test

# --
# Timeseries

def run_lengths_hist(T_train, T_test):
    # !! Super simple -- ignores values
    
    train_rls = [np.diff(np.where(np.diff(T_train[i]))[0]).astype(int) for i in range(len(T_train))]
    test_rls  = [np.diff(np.where(np.diff(T_test[i]))[0]).astype(int) for i in range(len(T_test))]
    
    thresh = np.percentile(np.hstack(train_rls), 95).astype(int)
    
    H_train = np.vstack([np.bincount(r[r <= thresh], minlength=thresh + 1) for r in train_rls])
    H_test  = np.vstack([np.bincount(r[r <= thresh], minlength=thresh + 1) for r in test_rls])
    
    return H_train, H_test

# --
# Sets

def set2hist(S_train, S_test, n_clusters=64, n_jobs=32, kmeans_sample=100000, batch_size=1000, verbose=False):
    
    S_train, S_test = list(S_train), list(S_test)
    n_train_obs, n_test_obs = len(S_train), len(S_test)
    
    train_offsets = np.cumsum([t.shape[0] for t in S_train])
    train_offsets = np.hstack([[0], train_offsets])
    
    test_offsets = np.cumsum([t.shape[0] for t in S_test])
    test_offsets = np.hstack([[0], test_offsets])
    
    dim = S_train[0].shape[1]
    assert len(set([t.shape[1] for t in S_train])) == 1
    assert len(set([t.shape[1] for t in S_test])) == 1
    
    S_train_flat, S_test_flat = np.vstack(S_train), np.vstack(S_test)
    T_all = np.vstack([S_train_flat, S_test_flat])
    
    kmeans_sample = min(kmeans_sample, T_all.shape[0])
    sel = np.random.choice(T_all.shape[0], kmeans_sample, replace=False)
    km  = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, verbose=verbose).fit(T_all[sel])
    
    cl_train = km.predict(S_train_flat)
    H_train = np.vstack([np.histogram(cl_train[train_offsets[i]:train_offsets[i+1]], bins=range(n_clusters + 1))[0] for i in range(n_train_obs)])
    
    cl_test = km.predict(S_test_flat)
    H_test  = np.vstack([np.histogram(cl_test[test_offsets[i]:test_offsets[i+1]], bins=range(n_clusters + 1))[0] for i in range(n_test_obs)])
    
    return H_train, H_test

