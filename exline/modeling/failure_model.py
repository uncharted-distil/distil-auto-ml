#!/usr/bin/env python

"""
    modeling/failure_model.y
"""

import numpy as np
from .metrics import metrics, classification_metrics, regression_metrics

class FailureModel:
    
    def __init__(self, target_metric):
        self.target_metric = target_metric
    
    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self
    
    def _object_detection_failure(self, X_test, y_test):
        y_test_act  = list(zip(list(X_test.values.squeeze()), y_test))
        y_test_pred = list(zip(list(X_test.values.squeeze()), np.random.choice(self._y_train, y_test.shape[0])))
        return metrics['objectDetectionAP'](y_test_act, y_test_pred)
    
    def score(self, X_test, y_test):
        if self.target_metric == 'objectDetectionAP':
            return self._object_detection_failure(X_test, y_test)
        
        if self.target_metric == 'objectDetectionAP':
            y_test = list(zip(list(X_test.values.squeeze()), y_test))
        
        if self.target_metric in ['meanSquaredError', 'rootMeanSquaredError', 'rootMeanSquaredErrorAvg']:
            best_guess = np.mean(self._y_train)
        
        elif self.target_metric in ['meanAbsoluteError']:
            best_guess = np.median(self._y_train)
        
        elif self.target_metric in classification_metrics:
            vals, cnts = np.unique(self._y_train, return_counts=True)
            best_guess = vals[cnts.argmax()]
            
        else:
            best_guess = np.random.choice(self._y_train, y_test.shape[0], replace=True)
        
        null_pred = np.repeat(best_guess, y_test.shape[0])
        return metrics[self.target_metric](y_test, null_pred)
