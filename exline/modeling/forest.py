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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import ParameterGrid

from .metrics import metrics, classification_metrics
from .helpers import tiebreaking_vote, adjust_f1_macro
from ..utils import parmap, maybe_subset

def EitherForestClassifier(**kwargs):
    kwargs = kwargs.copy()
    estimator = kwargs.pop('estimator')
    if estimator == 'ExtraTrees':
        return ExtraTreesClassifier(**kwargs)
    elif estimator == 'RandomForest':
        return RandomForestClassifier(**kwargs)
    else:
        raise Exception

def EitherForestRegressor(**kwargs):
    kwargs = kwargs.copy()
    estimator = kwargs.pop('estimator')
    if estimator == 'ExtraTrees':
        return ExtraTreesRegressor(**kwargs)
    elif estimator == 'RandomForest':
        return RandomForestRegressor(**kwargs)
    else:
        raise Exception


class ForestCV:
    classifier_param_grid = {
        "n_estimators"     : [32, 64, 128, 256, 512, 1024, 2048],
        "min_samples_leaf" : [1, 2, 4, 8, 16, 32],
        "class_weight"     : [None, "balanced"],
    }
    
    regression_param_grid = {
        "bootstrap"        : [True],
        "n_estimators"     : [32, 64, 128, 256, 512, 1024, 2048],
        "min_samples_leaf" : [2, 4, 8, 16, 32, 64],
    }
    
    def __init__(self, target_metric, subset=100000, final_subset=1500000, verbose=10, num_fits=1, inner_jobs=1, estimator=['RandomForest']):
        
        self.target_metric     = target_metric
        self.is_classification = target_metric in classification_metrics
        
        self.subset       = subset
        self.final_subset = final_subset
        self.verbose      = verbose
        self.num_fits     = num_fits
        self.inner_jobs   = inner_jobs
        self.outer_jobs   = 64
        self.estimator    = estimator
        
        self._models  = []
        self._y_train = None
    
    def fit(self, Xf_train, y_train, param_grid=None):
        self._y_train = y_train
        if self.is_classification:
            
            assert y_train.dtype == int
            assert y_train.min() == 0, 'may need to remap_labels'
            assert y_train.max() == len(set(y_train)) - 1, 'may need to remap_labels'
            
            print('ForestCV: self._fit_classifier', file=sys.stderr)
            self._models = [self._fit_classifier(Xf_train, y_train, param_grid=param_grid) for _ in range(self.num_fits)]
        else:
            print('ForestCV: self._fit_regressor', file=sys.stderr)
            self._models = [self._fit_regressor(Xf_train, y_train, param_grid=param_grid) for _ in range(self.num_fits)]
        
        return self
    
    def score(self, X, y):
        # !! May want to adjust F1 score.  ... but need to figure out when and whether it's helping
        return metrics[self.target_metric](y, self.predict(X))
    
    def predict(self, X):
        preds = [model.predict(X) for model in self._models]
        if self.is_classification:
            return tiebreaking_vote(np.vstack(preds), self._y_train)
        else:
            return np.stack(preds).mean(axis=0)
    
    def _fit_classifier(self, Xf_train, y_train, param_grid=None):
        assert self.estimator == ['RandomForest'] # !! DOn't want to accidentally do this
        
        self._y_train = y_train
        
        global _eval_grid_point
        
        if param_grid is None:
            param_grid = deepcopy(self.classifier_param_grid)
        
        param_grid.update({'estimator' : self.estimator}) # !!
        
        X, y = maybe_subset(Xf_train, y_train, n=self.subset)
        
        def _eval_grid_point(params):
            params['oob_score'] = True
            params['n_jobs']    = self.inner_jobs
            
            model     = EitherForestClassifier(**params).fit(X, y)
            score_oob = model.oob_decision_function_
            pred_oob  = model.classes_[score_oob.argmax(axis=-1)] # could vote better
            oob_score = metrics[self.target_metric](y, pred_oob)
            
            return {
                "params"  : params,
                "fitness" : oob_score,
            }
        
        # Run grid search
        self.results = parmap(_eval_grid_point, ParameterGrid(param_grid), verbose=self.verbose, n_jobs=self.outer_jobs)
        
        # Find best run
        best_run = sorted(self.results, key=lambda x: x['fitness'])[-1] # bigger is better
        self.best_params, self.best_fitness = best_run['params'], best_run['fitness']
        
        # Refit best model, possibly on more data
        self.best_params['n_jobs'] = self.outer_jobs
        X, y = maybe_subset(Xf_train, y_train, n=self.final_subset)
        return EitherForestClassifier(**self.best_params).fit(X, y)
    
    def _fit_regressor(self, Xf_train, y_train, param_grid=None):
        global _eval_grid_point
        
        if param_grid is None:
            param_grid = deepcopy(self.regression_param_grid)
        
        param_grid.update({'estimator' : self.estimator}) # !!
        
        X, y = maybe_subset(Xf_train, y_train, n=self.subset)
        
        def _eval_grid_point(params):
            params['oob_score'] = True
            params['n_jobs']    = self.inner_jobs
            
            model     = EitherForestRegressor(**params).fit(X, y)
            pred_oob  = model.oob_prediction_
            oob_score = metrics[self.target_metric](y, pred_oob)
            
            return {
                "params"  : params,
                "fitness" : oob_score,
            }
        
        # Run grid search
        self.results = parmap(_eval_grid_point, ParameterGrid(param_grid), verbose=self.verbose, n_jobs=self.outer_jobs)
        
        # Find best run
        best_run = sorted(self.results, key=lambda x: x['fitness'])[-1] # bigger is better
        self.best_params, self.best_fitness = best_run['params'], best_run['fitness']
        
        # Refit best model, possibly on more data
        self.best_params['n_jobs'] = self.outer_jobs
        X, y = maybe_subset(Xf_train, y_train, n=self.final_subset)
        return EitherForestRegressor(**self.best_params).fit(X, y)