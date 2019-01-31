#!/usr/bin/env python

"""
    exline/modeling/community_detection.py
"""

import sys
import numpy as np

from .metrics import metrics

class CommunityDetection:
    
    def __init__(self, target_metric, overlapping):
        self.target_metric = target_metric
        self.overlapping   = overlapping
    
    def fit(self, X_train, y_train, U_train=None):
        print('!! CommunityDetection: using null model', file=sys.stderr)
        return self
    
    def predict(self, X):
        return -np.arange(X.shape[0])
    
    @property
    def details(self):
        return {
            "null_model" : True
        }
