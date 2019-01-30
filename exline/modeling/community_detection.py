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
    
    def fit_score(self, graph, X_train, X_test, y_train, y_test):
        print('!! CommunityDetection: using null model', file=sys.stderr)
        
        # --
        # Null model, because I don't understand the problem
        
        null_pred  = -np.arange(y_test.shape[0])
        null_score = metrics[self.target_metric](y_test, null_pred)
        return null_score
