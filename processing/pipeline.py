import json
import logging
from typing import Tuple, Optional, List, Dict
import GPUtil
import sys

from d3m.container import dataset
from d3m import container, exceptions, runtime
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, pipeline, problem, pipeline_run
from d3m.metadata.pipeline import Pipeline, PlaceholderStep

from processing import metrics
from distil.primitives import utils as distil_utils
from distil.primitives.utils import CATEGORICALS

from processing import router

import config

from processing.pipelines import (clustering,
                                  collaborative_filtering,
                                  graph_matching,
                                  image,
                                  object_detection,
                                  question_answer,
                                  tabular,
                                  text,
                                  link_prediction,
                                  audio,
                                  vertex_nomination,
                                  vertex_classification,
                                  community_detection,
                                  timeseries_kanine,
                                  timeseries_var,
                                  timeseries_lstm_fcn,
                                  semisupervised_tabular)
                                #   data_augmentation_tabular) TODO: Looks like the data aug stuff has moved.

import utils

logger = logging.getLogger(__name__)


def create(dataset_doc_path: str, problem: dict, prepend: pipeline.Pipeline=None) -> Tuple[pipeline.Pipeline, container.Dataset]:

    # allow for use of GPU optimized pipelines
    gpu = _use_gpu()

    # Load dataset in the same way the d3m runtime will
    train_dataset = dataset.Dataset.load(dataset_doc_path)

    # If there isn't a placeholder this is a fully specified pipeline.  Return the pipeline unmodified along with the
    # dataset.
    if prepend and not [True for s in prepend.steps if isinstance(s, PlaceholderStep)]:
        return (prepend, train_dataset)

    # Load the dataset doc itself
    modified_path = dataset_doc_path.replace("file://", "")
    with open(modified_path) as json_file:
        dataset_doc = json.load(json_file)

    # extract metric from the problem
    protobuf_metric = problem['problem']['performance_metrics'][0]['metric']
    metric = metrics.translate_proto_metric(protobuf_metric)

    # determine type of pipeline required for dataset
    pipeline_type, pipeline_info = router.get_routing_info(dataset_doc, problem, metric)

    pipeline_type = pipeline_type.lower()

    pipeline: Pipeline = None
    if pipeline_type == 'table':
        pipeline = tabular.create_pipeline(metric)
    elif pipeline_type == 'graph_matching':
        pipeline = graph_matching.create_pipeline(metric)
    elif pipeline_type == 'timeseries_classification':
        if gpu:
            pipeline = timeseries_lstm_fcn.create_pipeline(metric)
        else:
            pipeline = timeseries_kanine.create_pipeline(metric)
    elif pipeline_type == 'question_answering':
        if gpu:
            pipeline = question_answer.create_pipeline(metric)
        else:
            pipeline = tabular.create_pipeline(metric)
    elif pipeline_type == 'text':
        pipeline = text.create_pipeline(metric)
    elif pipeline_type == 'image':
        pipeline = image.create_pipeline(metric)
    elif pipeline_type == 'object_detection':
        pipeline = object_detection.create_pipeline(metric)
    elif pipeline_type == 'audio':
       pipeline = audio.create_pipeline(metric)
    elif pipeline_type == 'collaborative_filtering':
        if gpu:
            pipeline = collaborative_filtering.create_pipeline(metric)
        else:
            pipeline = tabular.create_pipeline(metric)
    elif pipeline_type == 'vertex_nomination':
        pipeline = vertex_nomination.create_pipeline(metric)
    elif pipeline_type == 'vertex_classification':
        # force using vertex classification
        # TODO - should determine the graph data format
        pipeline = vertex_classification.create_pipeline(metric)
    elif pipeline_type == 'link_prediction':
        pipeline = link_prediction.create_pipeline(metric)
    elif pipeline_type == 'community_detection':
        pipeline = community_detection.create_pipeline(metric)
    elif pipeline_type == 'clustering':
        n_clusters = problem['inputs'][0]['targets'][0]['clusters_number']
        col_name = problem['inputs'][0]['targets'][0]['column_name']
        pipeline = clustering.create_pipeline(metric, num_clusters=n_clusters, cluster_col_name=col_name)
    elif pipeline_type == 'timeseries_forecasting':
        # VAR hyperparameters for period need to be tuned to get meaningful results so we're using regression
        # for now
        # pipeline = tabular.create_pipeline(metric)
        # the above was in the exline repo not sure what is the most up to date?
        pipeline = timeseries_var.create_pipeline(metric)
        # pipeline = timeseries_forecasting.create_pipeline(metric)
    elif pipeline_type == 'semisupervised_tabular':
        pipeline = semisupervised_tabular.create_pipeline(metric)
    elif pipeline_type == 'data_augmentation_tabular':
        pipeline = data_augmentation_tabular.create_pipeline(metric, dataset=train_dataset, keywords=pipeline_info)
    else:
        logger.error(f'Pipeline type [{pipeline_type}] is not yet supported.')
        return None, train_dataset

    # prepend to the base pipeline
    if prepend is not None:
        pipeline = _prepend_pipeline(pipeline, prepend)

    return pipeline, train_dataset


def fit(pipeline: pipeline.Pipeline, problem: problem.Problem, input_dataset: container.Dataset, is_standard_pipeline = True) -> Tuple[Optional[runtime.Runtime], Optional[runtime.Result]]:
    hyperparams = None
    random_seed = 0
    volumes_dir = config.D3MSTATICDIR

    fitted_runtime, _, result = runtime.fit(pipeline, [input_dataset], problem_description=problem, hyperparams=hyperparams, random_seed=random_seed,
                                            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING, runtime_environment=pipeline_run.RuntimeEnvironment(),
                                            is_standard_pipeline=is_standard_pipeline)

    if result.has_error():
        raise result.error

    return fitted_runtime, result


def produce(fitted_pipeline: runtime.Runtime, input_dataset: container.Dataset) -> runtime.Result:
    _, result = runtime.produce(fitted_pipeline, [input_dataset], expose_produced_outputs=True)
    if result.has_error():
        raise result.error
    return result


def is_fully_specified(prepend: pipeline.Pipeline) -> bool:
    # if there's a pipeline and it doesn't have a placeholder then its fully specified
    return prepend and not [True for s in prepend.steps if isinstance(s, PlaceholderStep)]


def _prepend_pipeline(base: pipeline.Pipeline, prepend: pipeline.Pipeline) -> pipeline.Pipeline:
    # find the placeholder node
    replace_index = -1
    for i, prepend_step in enumerate(prepend.steps):
        if isinstance(prepend_step, pipeline.PlaceholderStep):
            replace_index = i
            break

    if replace_index < 0:
        logger.warn(f'Failed to prepend pipeline {prepend.id} - continuing with base unmodified')
        return base

    # update prepend outputs to use those from base pipeline
    updated_outputs: List[Dict[str, str]] = []
    for base_output in base.outputs:
        # use the base pipeline output but increase the index to account for the appended base pipeline length
        base_ref = base_output['data'].split('.')
        updated_base_ref = f'{base_ref[0]}.{int(base_ref[1])+replace_index}.{base_ref[2]}'
        updated_outputs.append({'data': updated_base_ref, 'name': base_output['name']})
    prepend.outputs = updated_outputs

    # add each of the base steps, updating their references as necessary
    for base_idx, base_step in enumerate(base.steps):
        base_step.index = replace_index + base_idx
        if base_idx == 0:
            prepend.steps[base_step.index] = base_step
        else:
            prepend.steps.append(base_step)

        # update the inputs to account account for concatenation
        for arg_name, arg_val in base_step.arguments.items():
            arg_ref = arg_val['data'].split('.')
            if arg_ref[0] == 'inputs':
                # if this was taking inputs, take the last prepend node instead
                arg_val['data'] = f'steps.{int(arg_ref[1]) + replace_index-1}.produce'
            else:
                arg_val['data'] = f'{arg_ref[0]}.{int(arg_ref[1])+replace_index}.{arg_ref[2]}'

    return prepend

def _use_gpu() -> bool:
    # check for gpu presence - exception can be thrown when none available depending on
    # system config
    use_gpu = False
    try:
        gpus = GPUtil.getGPUs()
        logger.info(f'{len(gpus)} GPUs detected.  Requested GPU use is [{config.GPU}].')
        if (config.GPU == 'auto' or config.GPU == 'true') and len(gpus) > 0:
            use_gpu = True
        else:
            use_gpu = False
    except Exception as error:
            use_gpu = False
    logger.info(f'GPU enabled pipelines {use_gpu}')
    return use_gpu