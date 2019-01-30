#!/usr/bin/env python

"""
    exline/modeling/text_classification.py
"""

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from .metrics import metrics, classification_metrics, translate_d3m_metric

class TextClassifierCV:
    
    param_grid = {
        "vect__ngram_range"  : [(1, 1), (1, 2)],
        "vect__max_features" : [30000,],
        "cls__C"             : [float(xx) for xx in np.logspace(-3, 1, 1000)],
        "cls__class_weight"  : ['balanced', None],
    }
    
    def __init__(self, target_metric):
        assert target_metric in classification_metrics
        
        self.target_metric = target_metric
        self.scoring       = translate_d3m_metric(target_metric)
        self.n_jobs        = 32
    
    def fit(self, X_train, y_train, n_iter=256, n_folds=5, n_runs=1):
        
        self.model = RandomizedSearchCV(
            Pipeline([
                ('vect', TfidfVectorizer()),
                ('cls', LinearSVC())
            ]),
            n_iter=n_iter,
            param_distributions=self.param_grid,
            cv=RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_runs),
            scoring=self.scoring,
            iid=False,
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        self.model = self.model.fit(X_train, y_train)
        
        self.best_params  = self.model.best_params_
        self.best_fitness = self.model.cv_results_['mean_test_score'].max()
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return metrics[self.target_metric](y, self.predict(X))
