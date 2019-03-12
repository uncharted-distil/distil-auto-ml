import os
import sys
import json
import argparse
import numpy as np
from time import time
from typing import Dict, List
import logging

from d3m import container
from d3m.container import dataset
from d3m.metadata import base as metadata_base, problem
from d3m import runtime

from exline.io import load_problem
from exline.router import get_routing_info
from exline.d3m_util import PreprocessorFunctions, model_lookup
from exline.modeling.metrics import metrics
from exline.external import D3MDataset

from exline.primitives import tabular_pipeline


def exline_all(logger, dataset_doc_path: str, problem: dict) -> dict:
    # Load dataset in the same way the d3m runtime will
    train_dataset = dataset.Dataset.load(dataset_doc_path)

    # Temp hack to avoid metdata for now -
    modified_path = dataset_doc_path.replace("file://", "").replace("datasetDoc.json", "")
    column_info = D3MDataset(modified_path).get_learning_data_columns()
    column_types: Dict[str, type] = {}
    for c in column_info:
        col_name = c['colName']
        col_type = c['colType']
        if col_type == 'boolean':
            column_types[col_name] = bool
        elif col_type == 'real':
            column_types[col_name] = float
        elif col_type == 'integer':
            column_types[col_name] = int
        else:
            column_types[col_name] = str

    df = train_dataset[list(train_dataset.keys()).pop()]
    # TODO: set Target and Metric from request properly
    pipeline = tabular_pipeline.create_pipeline(df, column_types, 'Hall_of_Fame', 'f1Macro')

    inputs = [train_dataset]
    hyperparams = None
    random_seed = 0
    volumes_dir = None

    logger.info('Fitting...')
    # fitted_pipeline, predictions, fit_pipeline_run = runtime.fit()
    fitted_pipeline, _, _ = runtime.fit(
        pipeline, problem, inputs, hyperparams=hyperparams, random_seed=random_seed,
        volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
        runtime_environment=None
    )

    pipeline_json = fitted_pipeline.pipeline.to_json()
    from d3m.metadata.pipeline import Pipeline
    from d3m import container, utils
    PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)

    p = Pipeline(context=PipelineContext.TESTING).from_json(pipeline_json)
    logger.info(dir(p))

    #logger.info('Producing...')
    #outputs = runtime.produce(fitted_pipeline, inputs)
    #print(outputs)
    return pipeline_json, runtime