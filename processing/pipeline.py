import json
import logging
from typing import Tuple, Optional

from d3m.container import dataset
from d3m import container, exceptions, runtime
from d3m.metadata import base as metadata_base, pipeline, problem, pipeline_run

from distil.modeling import metrics
from processing import router

import config

from processing.pipelines import (clustering,
                                  collaborative_filtering,
                                  graph_matching,
                                  image,
                                  question_answer,
                                  tabular,
                                  text,
                                  timeseries_classification,
                                  link_prediction,
                                  audio,
                                  vertex_nomination,
                                  community_detection,
                                  timeseries_forecasting)


import utils

logger = logging.getLogger(__name__)


def create(dataset_doc_path: str, problem: dict, prepend: pipeline.Pipeline=None) -> Tuple[pipeline.Pipeline, container.Dataset]:
     # Load dataset in the same way the d3m runtime will
    train_dataset = dataset.Dataset.load(dataset_doc_path)

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

    if pipeline_type == 'table':
        pipeline = tabular.create_pipeline(metric)
    elif pipeline_type == 'graph_matching':
        pipeline = graph_matching.create_pipeline(metric)
    elif pipeline_type == 'timeseries_classification':
        pipeline = timeseries_classification.create_pipeline(metric)
    elif pipeline_type == 'question_answering':
        pipeline = question_answer.create_pipeline(metric)
    elif pipeline_type == 'text':
        pipeline = text.create_pipeline(metric)
    elif pipeline_type == 'image':
        pipeline = image.create_pipeline(metric)
    elif pipeline_type == 'audio':
       pipeline = audio.create_pipeline(metric)
    elif pipeline_type == 'collaborative_filtering':
        pipeline = collaborative_filtering.create_pipeline(metric)
    elif pipeline_type == 'vertex_nomination':
        pipeline = vertex_nomination.create_pipeline(metric)
    elif pipeline_type == 'link_prediction':
        pipeline = link_prediction.create_pipeline(metric)
    elif pipeline_type == 'community_detection':
        pipeline = community_detection.create_pipeline(metric)
    elif pipeline_type == 'clustering':
        n_clusters = problem['inputs'][0]['targets'][0]['clusters_number']
        col_name = problem['inputs'][0]['targets'][0]['column_name']
        pipeline = clustering.create_pipeline(metric, num_clusters=n_clusters, cluster_col_name=col_name)
    elif pipeline_type == 'timeseries_forecasting':
        pipeline = timeseries_forecasting.create_pipeline(metric)
    else:
        logger.error(f'Pipeline type [{pipeline_type}] is not yet supported.')
        return None, train_dataset

    # prepend to the base pipeline
    if prepend is not None:
        pipeline = _prepend_pipeline(pipeline, prepend)

    return pipeline, train_dataset


def fit(pipeline: pipeline.Pipeline, problem: problem.Problem, input_dataset: container.Dataset) -> Tuple[Optional[runtime.Runtime], Optional[runtime.Result]]:
    hyperparams = None
    random_seed = 0
    volumes_dir = config.D3MSTATICDIR

    fitted_runtime, _, result = runtime.fit(
        pipeline, problem, [input_dataset], hyperparams=hyperparams, random_seed=random_seed,
        volumes_dir=volumes_dir, context=metadata_base.Context.TESTING, runtime_environment=pipeline_run.RuntimeEnvironment()
    )

    if result.has_error():
        raise result.error

    return fitted_runtime, result


def produce(fitted_pipeline: runtime.Runtime, input_dataset: container.Dataset) -> container.DataFrame:
    predictions, result = runtime.produce(fitted_pipeline, [input_dataset])
    if result.has_error():
        raise result.error
    return predictions


def _prepend_pipeline(base: pipeline.Pipeline, prepend: pipeline.Pipeline) -> pipeline.Pipeline:
    # wrap pipeline in a sub pipeline - d3m core node replacement function doesn't work otherwise
    subpipeline = pipeline.SubpipelineStep(pipeline=base)

    # If there isn't a placeholder, return the prepended pipe
    if not [True for s in prepend.steps if isinstance(s, pipeline.PlaceholderStep)]:
        return prepend

    # find the placeholder node in the prepend and replace it with the base sub pipeline
    for i, step in enumerate(prepend.steps):
        if isinstance(step, pipeline.PlaceholderStep):
            # set inputs/outputs manually since the replace doesn't infer them
            for input_ref in step.inputs:
                subpipeline.add_input(input_ref)
            for output_id in step.outputs:
                subpipeline.add_output(output_id)

            prepend.replace_step(i, subpipeline)
            return prepend

    logger.warn(f'Failed to prepend pipeline {prepend.id} - continuing with base unmodified')
    return base
