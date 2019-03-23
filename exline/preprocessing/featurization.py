#!/usr/bin/env python

"""
    exline/preprocessing/featurization.py
"""

import sys
import numpy as np
import pandas as pd

from sklearn_pandas import DataFrameMapper as _DataFrameMapper
from sklearn_pandas import CategoricalImputer
from sklearn import impute, preprocessing

from .utils import MISSING_VALUE_INDICATOR, SINGLETON_INDICATOR
from .transformers import SVMTextEncoder, BinaryEncoder, enrich_dates

pd.options.mode.chained_assignment = None 

# --
# Helpers

def replace_singletons(X_train, X_test, keep_text=True):
    """ set values that only occur once to a special token """
    
    cols = list(X_train.columns)
    for c in cols:
        if X_train[c].dtype == np.object_:
            if not keep_text or not detect_text(X_train[c]):
                vcs        = pd.value_counts(list(X_train[c]) + list(X_test[c]))
                singletons = set(vcs[vcs == 1].index)
                if singletons:
                    X_train[c][X_train[c].isin(singletons)] = SINGLETON_INDICATOR
                    X_test[c][X_test[c].isin(singletons)]   = SINGLETON_INDICATOR
    
    return X_train, X_test


def detect_text(X, thresh=8):
    """ returns true if median entry has more than `thresh` tokens"""
    X = X[X.notnull()]
    n_toks = X.apply(lambda xx: len(str(xx).split(' '))).values
    return np.median(n_toks) >= thresh


# --
# Mapper

class DataFrameMapper:
    _possible_cat_modes = ['one_hot', 'binary']
    
    def __init__(self, target_metric, scale=False, cat_mode='one_hot', max_one_hot=16, max_binary=10000):
        assert cat_mode in self._possible_cat_modes
        
        self.target_metric = target_metric
        self.scale         = scale
        self.cat_mode      = cat_mode
        self.max_one_hot   = max_one_hot
        self.max_binary    = max_binary
        
        self._label_lookup = None
        self._mapper = None
    
    def pipeline(self, X_train, X_test, y_train):
        
        X_train, X_test = X_train.copy(), X_test.copy()
        
        # --
        # Prep input data
        
        X_train, X_test = enrich_dates(X_train, X_test)
        
        X_train, X_test = replace_singletons(X_train, X_test, keep_text=True)
        
        _ = self.fit(X_train)
        
        Xf_train, Xf_test = self.apply(X_train, X_test, y_train=y_train)
        
        return Xf_train, Xf_test
    
    def fit(self, X):
        maps = []
        for c in X.columns:
            if X[c].dtype == np.object_:
                uvals = list(set(X[c]))
                if len(uvals) == 1:
                    print('%s has 1 level -> skipping' % c, file=sys.stderr)
                    continue
                
                categories = [uvals + [MISSING_VALUE_INDICATOR]]
                
                if detect_text(X[c]):
                    print('----- %s has text detected -> using SVMTextEncoder' % c, file=sys.stderr)
                    cat_encoder = SVMTextEncoder()
                elif (self.cat_mode == 'one_hot') and (len(uvals) < self.max_one_hot):
                    # if one hot and small number of levels
                    cat_encoder = preprocessing.OneHotEncoder(sparse=False, categories=categories, handle_unknown='ignore')
                else:
                    # if large number of levels
                    print('%s has %d levels -> using BinaryEncoder' % (c, len(uvals)), file=sys.stderr)
                    cat_encoder = BinaryEncoder()
                
                maps.append(([c], [
                    CategoricalImputer(strategy='constant', fill_value=MISSING_VALUE_INDICATOR),
                    cat_encoder,
                ]))
            
            elif X[c].dtype in [float, int, bool]:
                
                transforms = [impute.SimpleImputer()]
                if self.scale:
                    raise Exception
                    transforms.append(preprocessing.StandardScaler())
                
                maps.append(([c], transforms))
                
                if X[c].isnull().any():
                    maps.append(([c], [impute.MissingIndicator()]))
            else:
                raise NotImplemented
        
        self._mapper = _DataFrameMapper(maps)
        return self
    
    def apply(self, X_train, X_test, y_train=None):
        Xf_train = self._mapper.fit_transform(X_train, y_train)
        Xf_test  = self._mapper.transform(X_test)
        
        # Drop columns w/ all zeros
        keep = (
            ((Xf_train != 0).sum(axis=0) != 0) & 
            ((Xf_test != 0).sum(axis=0) != 0)
        )
        Xf_train, Xf_test = Xf_train[:,keep], Xf_test[:,keep]
        
        return Xf_train, Xf_test
