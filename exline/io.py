#!/usr/bin/env python

"""
    exline/io.py
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy

from .external import D3MDS

# class EXProblem:
#     def __init__(self, prob_name, base_path):
        
#         self.d3mds = D3MDS(
#             os.path.join(base_path, prob_name, '%s_dataset' % prob_name),
#             os.path.join(base_path, prob_name, '%s_problem' % prob_name),
#         )
        
#         self.X_train = self.d3mds.get_train_data()
#         self.X_test  = self.d3mds.get_test_data()
#         self.y_train = self.d3mds.get_train_targets().squeeze()
#         self.y_test  = self.d3mds.get_test_targets().squeeze()
        
#         self.metric = self.d3mds.problem.get_performance_metrics()[0]['metric']


def load_problem(prob_name, base_path, return_d3mds=False, use_schema=False, strict=True):
    d3mds = D3MDS(
        os.path.join(base_path, prob_name, '%s_dataset' % prob_name),
        os.path.join(base_path, prob_name, '%s_problem' % prob_name),
    )
    
    X_train = d3mds.get_train_data()
    X_test  = d3mds.get_test_data()
    y_train = d3mds.get_train_targets().squeeze()
    y_test  = d3mds.get_test_targets().squeeze()
    
    
    if use_schema:
        print("use_schema", file=sys.stderr)
        for c in d3mds.dataset.get_learning_data_columns():
            col_type = c['colType']
            col_name = c['colName']
            if col_name not in X_train.columns:
                continue
            
            if col_type in ['categorical', 'string', 'boolean']:
                print("%s -> categorical" % col_name, file=sys.stderr)
                X_train[col_name] = X_train[col_name].astype(str)
                X_test[col_name]  = X_test[col_name].astype(str)
            elif col_type in ['real', 'integer']:
                print("%s: %s -> float" % (col_name, col_type), file=sys.stderr)
                X_train[col_name] = X_train[col_name].astype(np.float64)
                X_test[col_name]  = X_test[col_name].astype(np.float64)
            elif col_type in ['dateTime']:
                print('not preprocessing dateTime for now', file=sys.stderr)
            else:
                print('do nothing for unknown column type %s' % col_type, file=sys.stderr)
        
        # !! use role somehow?
    
    if strict:
        assert len(d3mds.problem.get_performance_metrics()) == 1
    
    ll_metric = d3mds.problem.get_performance_metrics()[0]['metric']
    
    score_path = os.path.join(base_path, prob_name, '%s_solution/scores.csv' % prob_name)
    ll_score = None
    if os.path.exists(score_path):
        try:
            ll_score = float(pd.read_csv(score_path).value)
        except:
            ll_score = pd.read_csv(score_path)
            ll_score = ll_score.set_index('metric').to_dict()
    
    return X_train, X_test, y_train, y_test, ll_metric, ll_score, d3mds


# --
# Timeseries

def maybe_truncate(T_train, T_test):
    # Truncate (!! Maybe it'd be better to pad...)
    ts_lengths = set([t.shape[0] for t in T_train] + [t.shape[0] for t in T_test])
    if len(ts_lengths) > 1:
        min_length = min(ts_lengths)
        print('maybe_truncate_sequences: truncating to %d' % min_length, file=sys.stderr)
        T_train = [t[-min_length:] for t in T_train]
        T_test  = [t[-min_length:] for t in T_test]
    
    T_train = np.vstack(T_train).astype(np.float64)
    T_test  = np.vstack(T_test).astype(np.float64)
    
    return T_train, T_test


class CollectionLoaders:
    @staticmethod
    def timeseries(paths):
        tmp = [pd.read_csv(p) for p in paths]
        assert np.all([t.shape[1] == 2 for t in tmp]) # assuming 1-d time series
        tmp = [t.values[:,1] for t in tmp]
        return tmp
    
    @staticmethod
    def text(paths):
        return [open(p).read() for p in paths]
    
    @staticmethod
    def table(paths):
        tmp = [pd.read_csv(p).values for p in paths]
        
        dims = [t.shape[1] for t in tmp]
        assert len(set(dims)) == 1
        
        return tmp


def load_ragged_collection(X_train, X_test, d3mds, collection_type='timeseries'):
    
    # --
    # Parse data resources
    
    resources = deepcopy(d3mds.dataset.dsDoc['dataResources'])
    
    # Get learningData resource description
    learning_resource = [r for r in resources if 'learningData.csv' in r['resPath']][0]
    resources.remove(learning_resource)
    
    # Get collection resource description
    collection_resource = [r for r in resources if r['resType'] == collection_type]
    assert len(collection_resource) == 1
    collection_resource = collection_resource[0]
    
    # Get column that links from learningData to collection
    ref_col = [c for c in learning_resource['columns'] if 'refersTo' in c.keys()]
    assert len(ref_col) == 1
    ref_col = ref_col[0]
    
    meta_cols = [c['colName'] for c in learning_resource['columns'] if (
        ('refersTo' not in c.keys()) and 
        (c['colName'] != 'd3mIndex') and 
        (c['colName'] in X_train.columns)
    )]
    
    # --
    # Load collection
    
    _loader = getattr(CollectionLoaders, collection_type)
    
    base_path = d3mds.dataset.dsHome
    
    train_paths = X_train[ref_col['colName']].apply(lambda x: os.path.join(base_path, collection_resource['resPath'], x))
    T_train = _loader(train_paths)
    
    test_paths  = X_test[ref_col['colName']].apply(lambda x: os.path.join(base_path, collection_resource['resPath'], x))
    T_test = _loader(test_paths)
    
    return T_train, T_test, meta_cols


def load_and_join(X_train, X_test, d3mds):
    
    X_train, X_test = X_train.copy(), X_test.copy()
    
    # --
    # Parse data resources
    
    resources = deepcopy(d3mds.dataset.dsDoc['dataResources'])
    
    # Get learningData resource description
    learning_resource = [r for r in resources if 'learningData.csv' in r['resPath']][0]
    resources.remove(learning_resource)
    
    # Get collection resource description
    collection_resources = [r for r in resources if (r['resType'] == 'table')]
    
    # Get columns that links from learningData to collection
    ref_cols = [c for c in learning_resource['columns'] if 'refersTo' in c.keys()]
    
    base_path = d3mds.dataset.dsHome
    
    # Load other tables
    cr_data = {}
    for cr in collection_resources:
        cr_data[cr['resID']] = pd.read_csv(os.path.join(base_path, cr['resPath']))
        
    for ref_col in ref_cols:
        X_train = pd.merge(X_train, cr_data[ref_col['refersTo']['resID']], 
            left_on=ref_col['colName'],
            right_on=ref_col['refersTo']['resObject']['columnName'],
            how='left'
        )
        X_test = pd.merge(X_test, cr_data[ref_col['refersTo']['resID']], 
            left_on=ref_col['colName'],
            right_on=ref_col['refersTo']['resObject']['columnName'],
            how='left'
        )
    
    # Additional columns
    meta_cols = [c['colName'] for c in learning_resource['columns'] if (
        ('refersTo' not in c.keys()) and 
        (c['colName'] != 'd3mIndex') and 
        (c['colName'] in X_train.columns)
    )]
    
    return X_train, X_test, meta_cols

# --
# Audio

import soundfile as sf

class WavInput:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate


def load_audio_file(path, row):
    if path[-3:] != 'wav':
        path += '.wav'
    
    info = sf.info(path)
    
    if ('start' in row) and ('end' in row):
        start = int(info.frames * (row.start / info.duration))
        end   = int(info.frames * (row.end / info.duration))
        wav_data, sample_rate = sf.read(path, start=start, stop=end, dtype='int16')
    else:
        wav_data, sample_rate = sf.read(path, dtype='int16')
    
    if len(wav_data.shape) == 1:
        wav_data = wav_data.reshape(-1, 1)
    
    if wav_data.shape[0] < sample_rate:
        wav_data = np.vstack([wav_data, np.zeros((sample_rate - wav_data.shape[0], wav_data.shape[1]), dtype='int16')])
    
    return WavInput(wav_data, sample_rate)

def load_audio(X_train, X_test, d3mds):
    print('!! load_audio: can only read WAV -- did this on command line', file=sys.stderr)
    
    base_path = d3mds.dataset.dsHome
    resources = d3mds.dataset.dsDoc['dataResources']
    
    # Get learningData resource description
    learning_resource = [r for r in resources if 'learningData.csv' in r['resPath']][0]
    
    # Get timeseries resource description
    audio_resource = [r for r in resources if r['resType'] == 'audio']
    assert len(audio_resource) == 1
    audio_resource = audio_resource[0]
    
    # Get column that links from learningData to timeseries
    ref_col = [c for c in learning_resource['columns'] if 'refersTo' in c.keys()]
    assert len(ref_col) == 1
    ref_col = ref_col[0]
    
    train_paths = X_train[ref_col['colName']].apply(lambda x: os.path.join(base_path, audio_resource['resPath'], x))
    A_train = [load_audio_file(path, row) for path, (_, row) in zip(train_paths, X_train.iterrows())]
    
    test_paths = X_test[ref_col['colName']].apply(lambda x: os.path.join(base_path, audio_resource['resPath'], x))
    A_test = [load_audio_file(path, row) for path, (_, row) in zip(test_paths, X_test.iterrows())]
    
    return A_train, A_test



# def load_ragged_timeseries(X, base_path):
#     c = 'timeseries_file' if 'timeseries_file' in X.columns else X.columns[0]
#     paths = X[c].apply(lambda x: os.path.join(base_path, 'timeseries', x))
#     return [pd.read_csv(p) for p in paths]


# def load_timeseries(X, base_path):
#     ts = load_ragged_timeseries(X, base_path)
#     assert len(set([len(tt) for tt in ts])) == 1 # assume all same length
#     return np.vstack(ts)


# def load_ragged_sets(X, base_path, colname=None):
#     colname = X.columns[0] if colname is None else colname
#     paths = X[colname].apply(lambda x: os.path.join(base_path, x))
#     return [pd.read_csv(p).values for p in paths]


# def load_sets(X, base_path, colname=None):
#     ts = load_ragged_sets(X, base_path, colname=colname)
#     assert len(set([t.shape for t in ts])) == 1
#     return np.stack(ts)

# --
# Graphs

def load_graphs(d3mds):
    Gs = {}
    for resource in d3mds.dataset.dsDoc['dataResources']:
        if resource['resType'] == 'graph':
            assert 'text/gml' in resource['resFormat']
            Gs[resource['resID']] = nx.read_gml(os.path.join(d3mds.dataset.dsHome, resource['resPath']))
            
    return Gs

# --
# Images


def prep_image_collection(X_train, X_test, d3mds, collection_type='image'):
    
    # --
    # Parse data resources
    
    resources = deepcopy(d3mds.dataset.dsDoc['dataResources'])
    
    # Get learningData resource description
    learning_resource = [r for r in resources if 'learningData.csv' in r['resPath']][0]
    resources.remove(learning_resource)
    
    # Get collection resource description
    collection_resource = [r for r in resources if r['resType'] == collection_type]
    assert len(collection_resource) == 1
    collection_resource = collection_resource[0]
    
    # Get column that links from learningData to collection
    ref_col = [c for c in learning_resource['columns'] if 'refersTo' in c.keys()]
    assert len(ref_col) == 1
    ref_col = ref_col[0]
    
    meta_cols = [c['colName'] for c in learning_resource['columns'] if (
        ('refersTo' not in c.keys()) and 
        (c['colName'] != 'd3mIndex') and 
        (c['colName'] in X_train.columns)
    )]
    
    # --
    # Load collection
    
    base_path = d3mds.dataset.dsHome
    
    train_paths = X_train[ref_col['colName']].apply(lambda x: os.path.join(base_path, collection_resource['resPath'], x))
    test_paths  = X_test[ref_col['colName']].apply(lambda x: os.path.join(base_path, collection_resource['resPath'], x))
    
    train_paths, test_paths = train_paths.values, test_paths.values
    
    return train_paths, test_paths, meta_cols


