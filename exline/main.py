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
from exline.modeling.pretrained_audio import AudiosetModel
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
    parser.add_argument('--prob-name',        type=str, default='solar_flare_2_regression')
    parser.add_argument('--base-path',        type=str, default='d3m_datasets/eval_datasets/LL0')
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

try:
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
        
        forest     = ForestCV(target_metric=ll_metric, **rparams)
        forest     = forest.fit(Xf_train, y_train)
        test_score = forest.score(Xf_test, y_test)
        
        _extra = {
            "cv_score"    : forest.best_fitness,
            "best_params" : forest.best_params,
            "num_fits"    : rparams["num_fits"],
        }
        
    elif route in ['multitable']:
        
        S_train, S_test, meta_cols = load_ragged_collection(X_train, X_test, d3mds, collection_type='table')
        H_train, H_test = set2hist(S_train, S_test)
        
        forest     = ForestCV(target_metric=ll_metric, **rparams)
        forest     = forest.fit(H_train, y_train)
        test_score = forest.score(H_test, y_test)
        
    elif route in ['question_answering']:
        
        X_train, X_test, meta_cols = load_and_join(X_train, X_test, d3mds)
        assert 'question' in X_train.columns
        assert 'sentence' in X_train.columns
        
        pc = BERTPairClassification(target_metric=ll_metric, **rparams)
        test_score = pc.fit_score(X_train, X_test, y_train, y_test)
        
    elif route in ['text']:
        
        S_train, S_test, meta_cols = load_ragged_collection(X_train, X_test, d3mds, collection_type='text')
        
        tc = TextClassifierCV(target_metric=ll_metric)
        tc = tc.fit(S_train, y_train)
        
        test_score = tc.score(S_test, y_test)
        
        _extra = {
            "best_params"  : tc.best_params,
            "best_fitness" : tc.best_fitness,
        }
        
    elif route in ['timeseries']:
        
        T_train, T_test, meta_cols = load_ragged_collection(X_train, X_test, d3mds, collection_type='timeseries')
        
        # Detect and and handle sparse timeseries
        if (np.hstack(T_train) == 0).mean() > 0.5:
            print('!! sparse timeseries', file=sys.stderr)
            T_train, T_test = run_lengths_hist(T_train, T_test)
            _ = rparams['metrics'].remove('dtw')
        
        # Detect and handle timeseries of different lengths
        T_train, T_test = maybe_truncate(T_train, T_test)
        
        # Prediction w/ time series only
        neighbors = NeighborsCV(target_metric=ll_metric, **rparams)
        neighbors = neighbors.fit(T_train, T_test, y_train, y_test)
        
        test_score = neighbors.score(T_test, y_test, y_train=y_train) # y_train for tiebreaking
        
        _extra = {
            "neighbors_fitness" : dict([(str(k), v) for k, v in neighbors.fitness.items()])
        }
        
        if meta_cols:
            E_train, E_test = X_train[meta_cols], X_test[meta_cols]
            
            df_mapper = DataFrameMapper(target_metric=ll_metric)
            Ef_train, Ef_test, y_train, y_test = df_mapper.pipeline(E_train, E_test, y_train, y_test)
            
            forest = ForestCV(target_metric=ll_metric)
            forest = forest.fit(Ef_train, y_train)
            
            extra_test_score = forest.score(Ef_test, y_test)
            
            _extra.update({
                "rf_meta_cols" : {
                    "cv_score"    : forest.best_fitness,
                    "best_params" : forest.best_params,
                }
            })
        
    elif route in ['audio']:
        
        A_train, A_test = load_audio(X_train, X_test, d3mds)
        
        ac = AudiosetModel(target_metric=ll_metric)
        ac = ac.fit(A_train, y_train)
        
        test_score = ac.score(A_test, y_test)
        
    elif route in ['graph_matching']:
        
        graphs = load_graphs(d3mds)
        assert len(graphs) == 2
        
        sgm = SGMGraphMatcher(target_metric=ll_metric)
        test_score = sgm.fit_score(graphs, X_train, X_test, y_train, y_test)
        
        _extra = {
            "train_acc"      : sgm.train_acc,
            "null_train_acc" : sgm.null_train_acc,
            
            "test_acc"       : sgm.test_acc,
            "null_test_acc"  : sgm.null_test_acc,
        }
        
    elif route in ['vertex_nomination']:
        
        graphs = load_graphs(d3mds)
        assert len(graphs) == 1
        graph = list(graphs.values())[0]
        
        vn = VertexNominationCV(target_metric=ll_metric)
        test_score = vn.fit_score(graph, X_train, X_test, y_train, y_test)
        
    elif route in ['collaborative_filtering']:
        
        cf = SGDCollaborativeFilter(target_metric=ll_metric)
        test_score = cf.fit_score(X_train, X_test, y_train, y_test)
        
    elif route in ['link_prediction']:
        
        graphs = load_graphs(d3mds)
        assert len(graphs) == 1
        graph = list(graphs.values())[0]
        
        rlp = RescalLinkPrediction(target_metric=ll_metric)
        test_score = rlp.fit_score(graph, X_train, X_test, y_train, y_test)
        
        # !! This should also be routed to something like
        # SGDCollaborativeFilter, w/ non-exclusive binary outputs
        
    elif route in ['community_detection']:
        
        graphs = load_graphs(d3mds)
        assert len(graphs) == 1
        graph = list(graphs.values())[0]
        
        cd = CommunityDetection(target_metric=ll_metric, **rparams)
        test_score = cd.fit_score(graph, X_train, X_test, y_train, y_test)
        
        _extra = {
            "null_model" : True
        }
        
    elif route in ['clustering']:
        
        # !! Should do preprocessing
        
        cl = ClusteringCV(target_metric=ll_metric, **rparams)
        test_score = cl.fit_score(X_train, X_test, y_train, y_test)
        
    elif route in ['image']:
        
        paths_train, paths_test, meta_cols = prep_image_collection(X_train, X_test, d3mds)
        
        cnn = FixedCNNForest(target_metric=ll_metric, **rparams)
        test_score = cnn.fit_score(paths_train, paths_test, y_train, y_test)
        
    else:
        raise Exception('no route triggered')
        
except:
    print('!!! CATCHING ERROR !!!', file=sys.stderr)
    
    fm = FailureModel(target_metric=ll_metric)
    fm = fm.fit(X_train, y_train)
    test_score = fm.score(X_test, y_test)
    
    _extra = {
        "FailureModel" : True,
    }


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

# Save results
results_dir = os.path.join('results', os.path.basename(os.path.dirname(args.base_path)))
os.makedirs(results_dir, exist_ok=True)

result_path = os.path.join(results_dir, args.prob_name)
open(result_path, 'w').write(json.dumps(res) + '\n')
