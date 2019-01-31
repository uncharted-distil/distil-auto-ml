#!/usr/bin/env python

"""
    exline/router.py
"""

import sys
from copy import deepcopy
from .modeling.metrics import classification_metrics, regression_metrics

SMALL_DATASET_THRESH = 2000

# --
# Detect semantic problem type

def get_resource_types(d3mds):
    return sorted([dr['resType'] for dr in d3mds.dataset.dsDoc['dataResources']])

def is_tabular(d3mds):
    resource_types = get_resource_types(d3mds)
    task_type = d3mds.problem.prDoc['about']['taskType']
    
    if task_type not in ['regression', 'classification', 'timeSeriesForecasting']:
        return False
    elif resource_types == ['table']:
        return True
    elif set(resource_types) == {'table'} and len(resource_types) > 2 and not is_question_answering(d3mds):
        print((
            "is_tabular: found more than two tables, which we can't handle.  falling back to single table"
        ), file=sys.stderr)
        return True
    else:
        return False

def is_multitable(d3mds):
    return ['table', 'table'] == get_resource_types(d3mds)

def is_timeseries(d3mds):
    return ['table', 'timeseries'] == get_resource_types(d3mds)

def is_question_answering(d3mds):
    res_paths = sorted([r['resPath'] for r in d3mds.dataset.dsDoc['dataResources']])
    return res_paths == ['tables/learningData.csv', 'tables/questions.csv', 'tables/sentences.csv', 'tables/vocabulary.csv']

def is_audio(d3mds):
    return 'audio' in get_resource_types(d3mds)

def is_image(d3mds):
    return 'image' in get_resource_types(d3mds)

def is_graph_matching(d3mds):
    return d3mds.problem.prDoc['about']['taskType'] == 'graphMatching'

def is_community_detection(d3mds):
    return d3mds.problem.prDoc['about']['taskType'] == 'communityDetection'

def is_clustering(d3mds):
    return d3mds.problem.prDoc['about']['taskType'] == 'clustering'

def is_vertex_nomination(d3mds):
    return d3mds.problem.prDoc['about']['taskType'] == 'vertexNomination'

def is_collaborative_filtering(d3mds):
    return d3mds.problem.prDoc['about']['taskType'] == 'collaborativeFiltering'

def is_link_prediction(d3mds):
    return d3mds.problem.prDoc['about']['taskType'] == 'linkPrediction'

def is_text(d3mds):
    return ['table', 'text'] == get_resource_types(d3mds)

# --
# Routing

def get_routing_info(X_train, X_test, y_train, y_test, metric, d3mds):
    
    # Shouldn't evaluate these in serial -- should do in parallel, then check for
    # conflicts
    
    if is_tabular(d3mds):
        return 'table', {
            "num_fits"  : 5 if X_train.shape[0] < SMALL_DATASET_THRESH else 1,
        }
    
    elif is_multitable(d3mds):
        
        resources = deepcopy(d3mds.dataset.dsDoc['dataResources'])
        learning_resource = [r for r in resources if 'learningData.csv' in r['resPath']][0]
        resources.remove(learning_resource)
        
        assert len(resources) == 1
        other_resource = resources[0]
        is_table       = other_resource['resType'] == 'table'
        is_collection  = other_resource['isCollection']
        
        if is_table and is_collection:
            return 'multitable', {}
        else:
            print((
                'get_routing_info: is_multitable, but other_resource is not a collection\n'
                ' falling back to table (eg. ignoring resources)'
            ), file=sys.stderr)
            
            return 'table', {
                "num_fits"  : 5 if X_train.shape[0] < SMALL_DATASET_THRESH else 1,
            }
    
    elif is_question_answering(d3mds):
        return 'question_answering', {}
    
    elif is_text(d3mds):
        return 'text', {}
    
    elif is_image(d3mds):
        return 'image', {}
    
    elif is_clustering(d3mds):
        
        resources = d3mds.dataset.dsDoc['dataResources']
        assert len(resources) == 1
        learning_resource = resources[0]
        
        # !! Not sure if I'm supposed to be looking at this
        n_clusters = d3mds.problem.prDoc['inputs']['data'][0]['targets'][0]['numClusters']
        
        all_float = set([r['colType'] for r in learning_resource['columns'] if 
            ('suggestedTarget' not in r['role']) and
            ('d3mIndex' != r['colName'])]) == {'real'}
        
        return 'clustering', {
            "n_clusters" : n_clusters,
            "all_float"  : all_float,
        }
    
    elif is_timeseries(d3mds):
        return 'timeseries', {
            "metrics"       : ['euclidean', 'cityblock', 'dtw'],
            "diffusion"     : metric in classification_metrics,
            "forest"        : True,
            "ensemble_size" : 3,
        }
    
    elif is_audio(d3mds):
        return 'audio', {}
    
    elif is_graph_matching(d3mds):
        return 'graph_matching', {}
    
    elif is_vertex_nomination(d3mds):
        return 'vertex_nomination', {}
    
    elif is_collaborative_filtering(d3mds):
        return 'collaborative_filtering', {}
        
    elif is_link_prediction(d3mds):
        return 'link_prediction', {}
    
    elif is_community_detection(d3mds):
        assert d3mds.problem.prDoc['about']['taskSubType'] == 'nonOverlapping'
        return 'community_detection', {
            'overlapping' : d3mds.problem.prDoc['about']['taskSubType'] != 'nonOverlapping',
        }
    
    else:
        raise Exception('!! router failed on problem')
