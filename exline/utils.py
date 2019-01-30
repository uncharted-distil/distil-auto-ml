#!/usr/bin/env python

"""
    exline/utils.py
"""

import numpy as np
from joblib import Parallel, delayed

def maybe_subset(X, y, n):
    if (n > 0) and (n < X.shape[0]):
        sel = np.sort(np.random.choice(X.shape[0], n, replace=False))
        return X[sel], y[sel]
    else:
        return X, y

def parmap(fn, x, n_jobs=64, backend='multiprocessing', verbose=1, **kwargs):
    jobs = [delayed(fn)(xx, **kwargs) for xx in x]
    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(jobs)
