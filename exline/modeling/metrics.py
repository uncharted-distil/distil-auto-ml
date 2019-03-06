#!/usr/bin/env python

"""
    exline/modeling/metrics

    !! Metrics are signed so that bigger is always better
"""

import sys
import numpy as np
from sklearn import metrics as sklearn_metrics
#from external import objectDetectionAP

metrics = {

    # classification
    'f1Macro'                     : lambda act, pred: sklearn_metrics.f1_score(act, pred, average='macro'),
    'f1'                          : lambda act, pred: sklearn_metrics.f1_score(act, pred),
    'accuracy'                    : lambda act, pred: sklearn_metrics.accuracy_score(act, pred),

    # regression
    'meanSquaredError'            : lambda act, pred: -1.0 * sklearn_metrics.mean_squared_error(act, pred),
    'meanAbsoluteError'           : lambda act, pred: -1.0 * sklearn_metrics.mean_absolute_error(act, pred),
    'rootMeanSquaredError'        : lambda act, pred: -1.0 * np.sqrt(sklearn_metrics.mean_squared_error(act, pred)),
    'rootMeanSquaredErrorAvg'     : lambda act, pred: -1.0 * np.sqrt(sklearn_metrics.mean_squared_error(act, pred)),

    # clustering
    'normalizedMutualInformation' : sklearn_metrics.normalized_mutual_info_score,

    # object detection
    #'objectDetectionAP' : lambda act, pred: objectDetectionAP(act, pred)[-1],
}

classification_metrics = set([
    'f1Macro',
    'f1',
    'accuracy',
])

regression_metrics = set([
    'meanSquaredError',
    'meanAbsoluteError',
    'rootMeanSquaredError',
    'rootMeanSquaredErrorAvg',
])

clustering_metrics = set([
    'normalizedMutualInformation',
])


def translate_d3m_metric(metric):
    if metric in ['rootMeanSquaredError', 'rootMeanSquaredErrorAvg']:
        print('translate_d3m_metric: metric=%s -> right ranking, but wrong values' % translate_d3m_metric, file=sys.stderr)

    lookup = {
        'f1Macro'              : 'f1_macro',
        'f1'                   : 'f1',
        'accuracy'             : 'accuracy',

        'meanSquaredError'        : 'neg_mean_squared_error',
        'rootMeanSquaredError'    : 'neg_mean_squared_error', # wrong values, but right ranking
        'rootMeanSquaredErrorAvg' : 'neg_mean_squared_error', # wrong values, but right ranking
    }
    assert metric in lookup, '%s not in lookup' % metric
    return lookup[metric]
