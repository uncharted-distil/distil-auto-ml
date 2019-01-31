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
from exline.d3m import PreprocessorFunctions, model_lookup

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob-name', type=str, default='185_baseball')
    parser.add_argument('--base-path', type=str, default='d3m_datasets/seed_datasets_current/')
    parser.add_argument('--seed',      type=int, default=456)
    return parser.parse_args()

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
# Route problem

route, hparams = get_routing_info(X_train, X_test, y_train, ll_metric, d3mds)

print('----------------------------------', file=sys.stderr)
print('prob_name:     %s' % args.prob_name, file=sys.stderr)
print('target_metric: %s' % ll_metric,      file=sys.stderr)
print('router:        %s' % route,          file=sys.stderr)
print('hparams:       %s' % str(hparams),   file=sys.stderr)
print('----------------------------------', file=sys.stderr)

# --
# Run

prep_fn   = getattr(PreprocessorFunctions, route)
model_cls = model_lookup[route]

Xf_train, Xf_test, U_train, hparams = prep_fn(X_train, X_test, y_train, d3mds, hparams)

model      = model_cls(target_metric=ll_metric, **hparams)
model      = model.fit(Xf_train, y_train, U_train)
pred_test  = model.predict(Xf_test)
test_score = metrics[ll_metric](y_test, pred_test)

# --
# Log

result = {
    "prob_name"     : args.prob_name,
    "ll_metric"     : ll_metric,
    "ll_score"      : ll_score, 
    "test_score"    : test_score,
    "elapsed"       : time() - t,
    "model_details" : model.details,
}

if not args.no_print_results:
    print(json.dumps(res))

if args.base_path[-1] != '/':
    args.base_path += '/'

# Save results
results_dir = os.path.join('results', os.path.basename(os.path.dirname(args.base_path)))
os.makedirs(results_dir, exist_ok=True)

result_path = os.path.join(results_dir, args.prob_name)
open(result_path, 'w').write(json.dumps(res) + '\n')
