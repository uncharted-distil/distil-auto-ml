#!/usr/bin/env python

"""
    exline/main.py
"""

import os
import sys
import json
import argparse
import numpy as np
from time import time

from exline.io import load_problem
from exline.router import get_routing_info
from exline.d3m import PreprocessorFunctions, model_lookup
from exline.modeling.metrics import metrics

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob-name', type=str, default='185_baseball')
    parser.add_argument('--base-path', type=str, default='d3m_datasets/seed_datasets_current/')
    parser.add_argument('--seed',      type=int, default=456)
    
    args = parser.parse_args()
    
    if args.base_path[-1] != '/':
        args.base_path += '/'
    
    return args


args = parse_args()
np.random.seed(args.seed)
t = time()

# --
# Load problem

X_train, X_test, y_train, y_test, ll_metric, ll_score, d3mds = load_problem(
    prob_name=args.prob_name,
    base_path=args.base_path,
)

# --
# Pick pipeline

route, hparams = get_routing_info(X_train, X_test, y_train, ll_metric, d3mds)
prep_fn        = getattr(PreprocessorFunctions, route)
model_cls      = model_lookup[route]

print('-----------------------------------',        file=sys.stderr)
print('prob_name     : %s' % args.prob_name,        file=sys.stderr)
print('target_metric : %s' % ll_metric,             file=sys.stderr)
print('route         : %s' % route,                 file=sys.stderr)
print('prep_fn       : %s' % str(prep_fn),          file=sys.stderr)
print('model_cls     : %s' % str(model_cls),        file=sys.stderr)
print('hparams       : %s' % str(hparams),          file=sys.stderr)
print('num_obs       : %s' % str(X_train.shape[0]), file=sys.stderr)
print('-----------------------------------',        file=sys.stderr)

Xf_train, Xf_test, U_train, hparams = prep_fn(X_train, X_test, y_train, ll_metric, d3mds, hparams)

model      = model_cls(target_metric=ll_metric, **hparams)
model      = model.fit(Xf_train, y_train, U_train)
pred_test  = model.predict(Xf_test)
test_score = metrics[ll_metric](y_test, pred_test)

# --
# Log Results

results = {
    "prob_name"     : args.prob_name,
    "ll_metric"     : ll_metric,
    "ll_score"      : ll_score, 
    "test_score"    : test_score,
    "elapsed"       : time() - t,
    "model_details" : model.details,
}
print(json.dumps(results))

results_dir = os.path.join('results', os.path.basename(os.path.dirname(args.base_path)))
os.makedirs(results_dir, exist_ok=True)

result_path = os.path.join(results_dir, args.prob_name)
open(result_path, 'w').write(json.dumps(results) + '\n')
