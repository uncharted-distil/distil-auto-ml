#!/usr/bin/env python

"""
    exline/modeling/random_forest.py
"""

SUPRESS_WARNINGS = True
if SUPRESS_WARNINGS:
    import sys
    def warn(*args, **kwargs): pass
    
    import warnings
    warnings.warn = warn

import sys
import numpy as np
from copy import deepcopy

from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold, \
    RandomizedSearchCV

from .metrics import metrics, classification_metrics, translate_d3m_metric
from ..utils import parmap, maybe_subset

class SupportVectorCV:
    classifier_param_grid = {
            "C"            : [float(xx) for xx in np.logspace(-3, 1, 1000)],
            "gamma"        : [float(xx) for xx in np.logspace(-2, 1, 1000)],
            "degree"       : [1, 2, 3, 4],
            "kernel"       : ['linear', 'poly', 'rbf'],
            "shrinking"    : [False, True],
            "class_weight" : ['balanced', None],
            # normalize or not
    }
    
    def __init__(self, target_metric, verbose=10):
        
        self.target_metric     = target_metric
        self.sklearn_metric    = translate_d3m_metric(target_metric)
        self.is_classification = target_metric in classification_metrics
        assert self.is_classification
        
        self.n_splits = 5
        self.n_runs   = 1
        self.n_jobs   = 64
        self.n_iter   = 1024
    
    def fit(self, Xf_train, y_train, param_grid=None):
        if self.is_classification:
            assert y_train.dtype == int
            assert y_train.min() == 0, 'may need to remap_labels'
            assert y_train.max() == len(set(y_train)) - 1, 'may need to remap_labels'
            
            if param_grid is None:
                param_grid = deepcopy(self.classifier_param_grid)
            
            model = RandomizedSearchCV(
                SVC(),
                param_distributions=param_grid,
                n_iter=self.n_iter,
                cv=RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_runs),
                scoring=self.sklearn_metric,
                iid=False,
                n_jobs=self.n_jobs,
                refit=True
            )
            
            model = model.fit(Xf_train, y_train)
            
            self.model        = model
            self.best_params  = model.best_params_
            self.best_fitness = model.cv_results_['mean_test_score'].max()
        else:
            raise NotImplemented
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return metrics[self.target_metric](y, self.predict(X))
