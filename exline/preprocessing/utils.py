#!/usr/bin/env python

"""
    exline/preprocessing/utils.py
"""

import numpy as np

MISSING_VALUE_INDICATOR = '__miss_salt%d' % np.random.choice(int(1e6))
SINGLETON_INDICATOR     = '__sing_salt%d' % np.random.choice(int(1e6))

_remappable_metrics = ['f1Macro', 'accuracy']


def _remap_labels(y_train, y_test):
    ulab     = list(set(y_train))
    y_lookup = dict(zip(ulab, range(len(ulab))))
    y_train  = np.array([y_lookup.get(yy) for yy in y_train])
    y_test   = np.array([y_lookup.get(yy) for yy in y_test])
    return y_train, y_test, y_lookup


def prep_labels(y_train, y_test, target_metric):
    if target_metric in _remappable_metrics:
        y_train, y_test, _ =  _remap_labels(y_train, y_test)
    elif target_metric == 'f1':
        assert list(sorted(set(y_train))) == [0, 1]
        assert list(sorted(set(y_test)))  == [0, 1]
    
    return y_train, y_test