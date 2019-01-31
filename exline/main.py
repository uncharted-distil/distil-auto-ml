#!/usr/bin/env python

"""
    exline/main.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from time import time

from exline.io import (
    load_problem,
    load_ragged_collection,
    load_graphs,
    load_audio,
    maybe_truncate,
    prep_image_collection,
    load_and_join,
)
from exline.router import get_routing_info

from exline.preprocessing.utils import prep_labels
from exline.preprocessing.featurization import DataFrameMapper
from exline.preprocessing.transformers import run_lengths_hist, set2hist

from exline.modeling.forest import ForestCV
from exline.modeling.neighbors import NeighborsCV
from exline.modeling.sgm import SGMGraphMatcher
from exline.modeling.vertex_nomination import VertexNominationCV
from exline.modeling.collaborative_filtering import SGDCollaborativeFilter
from exline.modeling.link_prediction import RescalLinkPrediction
from exline.modeling.text_classification import TextClassifierCV
# from exline.modeling.pretrained_audio import AudiosetModel # !! Slow to import
from exline.modeling.community_detection import CommunityDetection
from exline.modeling.clustering import ClusteringCV
from exline.modeling.fixed_cnn import FixedCNNForest
from exline.modeling.bert_models import BERTPairClassification
from exline.modeling.failure_model import FailureModel

from exline.modeling.metrics import metrics, classification_metrics, regression_metrics

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob-name',        type=str, default='185_baseball')
    parser.add_argument('--base-path',        type=str, default='d3m_datasets/seed_datasets_current/')
    parser.add_argument('--seed',             type=int, default=456)
    parser.add_argument('--use-schema',       action='store_true')
    parser.add_argument('--no-print-results', action='store_true')
    parser.add_argument('--rparams',          type=str, default='{}')
    return parser.parse_args()


args = parse_args()
np.random.seed(args.seed)

prob_args = {
    "prob_name"    : args.prob_name,
    "base_path"    : args.base_path,
    "return_d3mds" : True,
    "use_schema"   : args.use_schema,
    "strict"       : True,
}

# --
# Load problem

X_train, X_test, y_train, y_test, ll_metric, ll_score, d3mds = load_problem(**prob_args)
t = time()

y_train, y_test = prep_labels(y_train, y_test, target_metric=ll_metric)

# --
# Run model

# try:
route, rparams = get_routing_info(X_train, X_test, y_train, y_test, ll_metric, ll_score, d3mds)
rparams.update(json.loads(args.rparams))

print('-' * 50, file=sys.stderr)
print('prob_name=%s' % args.prob_name, file=sys.stderr)
print('target_metric=%s' % ll_metric, file=sys.stderr)
print('router: %s' % route, file=sys.stderr)
print('rparams: %s' % str(rparams), file=sys.stderr)
print('-' * 50, file=sys.stderr)

_extra = {}
if route in ['table']:
    
    df_mapper = DataFrameMapper(target_metric=ll_metric)
    Xf_train, Xf_test, y_train, y_test = df_mapper.pipeline(X_train, X_test, y_train, y_test)
    
    model      = ForestCV(target_metric=ll_metric, **rparams)
    model      = model.fit(Xf_train, y_train)
    pred_test  = model.predict(Xf_test)
    test_score = metrics[ll_metric](y_test, pred_test)
    
    _extra = {
        "cv_score"    : model.best_fitness,
        "best_params" : model.best_params,
        "num_fits"    : rparams["num_fits"],
    }
    
elif route in ['multitable']:
    
    Xf_train, Xf_test, meta_cols = load_ragged_collection(X_train, X_test, d3mds, collection_type='table')
    Xf_train, Xf_test = set2hist(Xf_train, Xf_test)
    
    model      = ForestCV(target_metric=ll_metric, **rparams)
    model      = model.fit(Xf_train, y_train)
    pred_test  = model.predict(Xf_test)
    test_score = metrics[ll_metric](y_test, pred_test)
    
elif route in ['question_answering']:
    
    Xf_train, Xf_test, meta_cols = load_and_join(X_train, X_test, d3mds)
    assert 'question' in Xf_train.columns
    assert 'sentence' in Xf_train.columns
    
    model      = BERTPairClassification(target_metric=ll_metric, **rparams)
    model      = model.fit(Xf_train, y_train)
    pred_test  = model.predict(Xf_test)
    test_score = metrics[ll_metric](y_test, pred_test)
    
elif route in ['text']:
    
    Xf_train, Xf_test, _ = load_ragged_collection(X_train, X_test, d3mds, collection_type='text')
    
    model      = TextClassifierCV(target_metric=ll_metric)
    model      = model.fit(Xf_train, y_train)
    pred_test  = model.predict(Xf_test)
    test_score = metrics[ll_metric](y_test, pred_test)
    
    _extra = {
        "best_params"  : model.best_params,
        "best_fitness" : model.best_fitness,
    }
    
elif route in ['timeseries']:
    # !! More annoying to convert to fit/predict -- semi-supervised AND ensemble
    
    Xf_train, Xf_test, meta_cols = load_ragged_collection(X_train, X_test, d3mds, collection_type='timeseries')
    
    # Detect and and handle sparse timeseries
    if (np.hstack(Xf_train) == 0).mean() > 0.5:
        print('!! sparse timeseries', file=sys.stderr)
        Xf_train, Xf_test = run_lengths_hist(Xf_train, Xf_test)
        _ = rparams['metrics'].remove('dtw')
    
    # Detect and handle timeseries of different lengths
    Xf_train, Xf_test = maybe_truncate(Xf_train, Xf_test)
    
    # Prediction w/ time series only
    model      = NeighborsCV(target_metric=ll_metric, **rparams)
    model      = model.fit(Xf_train, Xf_test, y_train, y_test)
    test_score = model.score(Xf_test, y_test)
    
    _extra = {
        "neighbors_fitness" : dict([(str(k), v) for k, v in model.fitness.items()])
    }
    
    # if meta_cols:
    #     df_mapper = DataFrameMapper(target_metric=ll_metric)
    #     Xf_train, Xf_test, y_train, y_test = df_mapper.pipeline(
    #         X_train[meta_cols], X_test[meta_cols], y_train, y_test)
        
    #     extra_model      = ForestCV(target_metric=ll_metric)
    #     extra_model      = extra_model.fit(Xf_train, y_train)
    #     extra_test_score = extra_model.score(Xf_test, y_test)
        
    #     _extra.update({
    #         "rf_meta_cols" : {
    #             "cv_score"    : extra_model.best_fitness,
    #             "best_params" : extra_model.best_params,
    #         }
    #     })
    
elif route in ['audio']:
    from exline.modeling.pretrained_audio import AudiosetModel
    
    Xf_train, Xf_test = load_audio(X_train, X_test, d3mds)
    
    model      = AudiosetModel(target_metric=ll_metric)
    model      = model.fit(Xf_train, y_train)
    pred_test  = model.predict(Xf_test)
    test_score = metrics[ll_metric](y_test, pred_test)
    
elif route in ['graph_matching']:
    
    graphs = load_graphs(d3mds)
    assert len(graphs) == 2
    
    model      = SGMGraphMatcher(target_metric=ll_metric)
    model      = model.fit(graphs, X_train, y_train)
    pred_test  = model.predict(X_test)
    test_score = metrics[ll_metric](y_test, pred_test)
    
    _extra = {
        "train_acc"      : model.sgm_train_acc,
        "null_train_acc" : model.null_train_acc,
    }
    
elif route in ['vertex_nomination']:
    
    graphs = load_graphs(d3mds)
    assert len(graphs) == 1
    graph = list(graphs.values())[0]
    
    model      = VertexNominationCV(target_metric=ll_metric)
    model      = model.fit(graph, X_train, y_train)
    pred_test  = model.predict(X_test)
    test_score = metrics[ll_metric](y_test, pred_test)
    
elif route in ['collaborative_filtering']:
    
    model      = SGDCollaborativeFilter(target_metric=ll_metric)
    test_score = model.fit_score(X_train, X_test, y_train, y_test)
    
elif route in ['link_prediction']:
    
    graphs = load_graphs(d3mds)
    assert len(graphs) == 1
    graph = list(graphs.values())[0]
    
    model      = RescalLinkPrediction(target_metric=ll_metric)
    model      = model.fit(graph, X_train, y_train)
    pred_test  = model.predict(X_test)
    test_score = metrics[ll_metric](y_test, pred_test)
    
    # !! This should also be routed to something like
    # SGDCollaborativeFilter, w/ non-exclusive binary outputs
    
elif route in ['community_detection']:
    
    graphs = load_graphs(d3mds)
    assert len(graphs) == 1
    graph = list(graphs.values())[0]
    
    model      = CommunityDetection(target_metric=ll_metric, **rparams)
    model      = model.fit(graph, X_train, y_train)
    pred_test  = model.predict(X_test)
    test_score = metrics[ll_metric](y_test, pred_test)
    
    _extra = {
        "null_model" : True
    }
    
elif route in ['clustering']:
    # !! Should do preprocessing
    
    model      = ClusteringCV(target_metric=ll_metric, **rparams)
    model      = model.fit(X_train, y_train)
    pred_test  = model.predict(X_test)
    test_score = metrics[ll_metric](y_test, pred_test)
    
elif route in ['image']:
    
    Xf_train, Xf_test, meta_cols = prep_image_collection(X_train, X_test, d3mds)
    
    model      = FixedCNNForest(target_metric=ll_metric, **rparams)
    model      = model.fit(Xf_train, y_train)
    pred_test  = model.predict(Xf_test)
    test_score = metrics[ll_metric](y_test, pred_test)
    
else:
    raise Exception('no route triggered')

# except:
#     print('!!! CATCHING ERROR !!!', file=sys.stderr)
    
#     fm = FailureModel(target_metric=ll_metric)
#     fm = fm.fit(X_train, y_train)
#     test_score = fm.score(X_test, y_test)
    
#     _extra = {
#         "FailureModel" : True,
#     }


# --
# Log results

res = {
    "prob_name" : args.prob_name,
    "ll_metric" : ll_metric,
    "ll_score"  : ll_score, 
    
    "test_score" : test_score,
    
    "elapsed" : time() - t,
    
    "_extra" : _extra,
    "_misc"  : {
        "use_schema" : args.use_schema,
    }
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
