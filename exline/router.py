#!/usr/bin/env python

"""
    exline/router.py
"""

import sys
import logging
from copy import deepcopy
from .modeling.metrics import classification_metrics, regression_metrics
from typing import Sequence, Tuple
from d3m.metadata import problem

logger = logging.getLogger(__name__)

SMALL_DATASET_THRESH = 2000

# --
# Detect semantic problem type

def get_resource_types(dataset_doc: dict) -> Sequence[str]:
    return sorted([dr['resType'] for dr in dataset_doc['dataResources']])

def is_tabular(dataset_doc: dict, problem_desc: dict) -> bool:
    resource_types = get_resource_types(dataset_doc)
    task_type = problem_desc['problem']['task_type']

    if task_type not in [problem.TaskType.REGRESSION, problem.TaskType.CLASSIFICATION, problem.TaskType.TIME_SERIES_FORECASTING]:
        return False
    elif resource_types == ['table']:
        return True
    elif set(resource_types) == {'table'} and len(resource_types) > 2 and not is_question_answering(dataset_doc):
        print((
            "is_tabular: found more than two tables, which we can't handle.  falling back to single table"
        ), file=sys.stderr)
        return True
    else:
        return False

def is_multitable(dataset_doc: dict) -> bool:
    return ['table', 'table'] == get_resource_types(dataset_doc)

def is_timeseries(dataset_doc: dict) -> bool:
    return ['table', 'timeseries'] == get_resource_types(dataset_doc)

def is_question_answering(dataset_doc: dict) -> bool:
    res_paths = sorted([r['resPath'] for r in dataset_doc['dataResources']])
    return res_paths == ['tables/learningData.csv', 'tables/questions.csv', 'tables/sentences.csv', 'tables/vocabulary.csv']

def is_audio(dataset_doc: dict) -> bool:
    return 'audio' in get_resource_types(dataset_doc)

def is_image(dataset_doc: dict) -> bool:
    return 'image' in get_resource_types(dataset_doc)

def is_graph_matching(problem: dict) -> bool:
    return problem_desc['problem']['task_type'] == problem.TaskType.GRAPH_MATCHING

def is_community_detection(problem: dict) -> bool:
    return problem_desc['problem']['task_type'] == problem.TaskType.COMMUNITY_DETECTION

def is_clustering(problem: dict) -> bool:
    return problem_desc['problem']['task_type'] == problem.TaskType.CLUSTERING

def is_vertex_nomination(problem: dict) -> bool:
    return problem_desc['problem']['task_type'] == problem.TaskType.VERTEX_NOMINATION

def is_collaborative_filtering(problem: dict) -> bool:
    return problem_desc['problem']['task_type'] == problem.TaskType.COLLABORATIVE_FILTERING

def is_link_prediction(problem: dict) -> bool:
    return problem_desc['problem']['task_type'] == problem.TaskType.LINK_PREDICTION

def is_text(dataset_doc: dict) -> bool:
    return ['table', 'text'] == get_resource_types(dataset_doc)

# --
# Routing

def get_routing_info(dataset_doc: dict, problem: dict, metric: str) -> Tuple[str, dict]:

    # Shouldn't evaluate these in serial -- should do in parallel, then check for
    # conflicts

    if is_tabular(dataset_doc, problem):
        return 'table', {
            'num_fits': 1
            # "num_fits"  : 5 if X_train.shape[0] < SMALL_DATASET_THRESH else 1,
        }

    elif is_multitable(dataset_doc):

        resources = deepcopy(dataset_doc['dataResources'])
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
                #"num_fits"  : 5 if X_train.shape[0] < SMALL_DATASET_THRESH else 1,
                'num_fits': 1
            }

    elif is_question_answering(dataset_doc):
        return 'question_answering', {}

    elif is_text(dataset_doc):
        return 'text', {}

    elif is_image(dataset_doc):
        return 'image', {}

    elif is_clustering(problem):

        resources = dataset_doc['dataResources']
        assert len(resources) == 1
        learning_resource = resources[0]

        # !! Not sure if I'm supposed to be looking at this
        n_clusters = problem['inputs']['data'][0]['targets'][0]['numClusters']

        all_float = set([r['colType'] for r in learning_resource['columns'] if
            ('suggestedTarget' not in r['role']) and
            ('d3mIndex' != r['colName'])]) == {'real'}

        return 'clustering', {
            "n_clusters" : n_clusters,
            "all_float"  : all_float,
        }

    elif is_timeseries(dataset_doc):
        return 'timeseries', {
            "metrics"       : ['euclidean', 'cityblock', 'dtw'],
            "diffusion"     : metric in classification_metrics,
            "forest"        : True,
            "ensemble_size" : 3,
        }

    elif is_audio(dataset_doc):
        return 'audio', {}

    elif is_graph_matching(problem):
        return 'graph_matching', {}

    elif is_vertex_nomination(problem):
        return 'vertex_nomination', {}

    elif is_collaborative_filtering(problem):
        return 'collaborative_filtering', {}

    elif is_link_prediction(problem):
        return 'link_prediction', {}

    elif is_community_detection(problem):
        assert problem['problem']['taskSubType'] == 'nonOverlapping'
        return 'community_detection', {
            'overlapping' : problem['taskSubType'] != 'nonOverlapping',
        }

    else:
        raise Exception('!! router failed on problem')
