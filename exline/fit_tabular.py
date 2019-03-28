import os
import sys
import json
import argparse
import numpy as np
from time import time
from typing import Dict, List, Set
import logging
import re

from d3m import container, exceptions, runtime
from d3m.container import dataset
from d3m.metadata import base as metadata_base, problem, pipeline
from d3m import runtime

from exline.io import load_problem
from exline.router import get_routing_info
from exline.d3m_util import PreprocessorFunctions, model_lookup
from exline.modeling.metrics import metrics, translate_proto_metric
from exline.external import D3MDataset

from exline.primitives import tabular_pipeline


logger = logging.getLogger('exline')

def fit(dataset_doc_path: str, problem: dict, prepend: pipeline.Pipeline=None) -> runtime.Runtime:

    # Load dataset in the same way the d3m runtime will
    train_dataset = dataset.Dataset.load(dataset_doc_path)

    # Temp hack to avoid metdata for now -
    modified_path = dataset_doc_path.replace("file://", "").replace("datasetDoc.json", "")

    # extract target column and metric from the problem
    df = train_dataset[list(train_dataset.keys()).pop()]
    protobuf_metric = problem['problem']['performanceMetrics'][0]['metric']
    metric = translate_proto_metric(protobuf_metric)

    pipeline = tabular_pipeline.create_pipeline(df, metric)

    # prepend to the base pipeline
    if prepend is not None:
        pipeline = prepend_pipeline(pipeline, prepend)
        logger.warn(pipeline)

    inputs = [train_dataset]
    hyperparams = None
    random_seed = 0
    volumes_dir = None

    fitted_pipeline, _, _ = runtime.fit(
        pipeline, problem, inputs, hyperparams=hyperparams, random_seed=random_seed,
        volumes_dir=volumes_dir, context=metadata_base.Context.TESTING
    )

    return fitted_pipeline

def prepend_pipeline(base: pipeline.Pipeline, prepend: pipeline.Pipeline) -> pipeline.Pipeline:
    # wrap pipeline in a sub pipeline - d3m core node replacement function doesn't work otherwise
    subpipeline = pipeline.SubpipelineStep(pipeline=base)

    # find the placeholder node in the prepend and replace it with the base sub pipeline
    for i, step in enumerate(prepend.steps):
        if isinstance(step, pipeline.PlaceholderStep):
            # set inputs/outputs manually since the replace doesn't infer them
            for input_ref in step.get_input_data_references():
                subpipeline.add_input(input_ref)
            for output_id in step.outputs:
                subpipeline.add_output(output_id)

            prepend.replace_step(i, subpipeline)
            return prepend

    logger.warn(f'Failed to prepend pipeline {prepend.id} - continuing with base unmodified')
    return base