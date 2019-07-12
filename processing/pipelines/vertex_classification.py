import sys
from typing import List, Dict, Any, Tuple, Set
import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from sri.graph.vertex_classification import VertexClassificationParser
from sri.psl.vertex_classification import VertexClassification

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive


PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)


def create_pipeline(metric: str) -> Pipeline:

    # create the basic pipeline
    vertex_classification_pipeline = Pipeline(context=PipelineContext.TESTING)
    vertex_classification_pipeline.add_input(name='inputs')

    # step 0 - extract the graphs
    step = PrimitiveStep(primitive_description=VertexClassificationParser.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    vertex_classification_pipeline.add_step(step)

    # step 1 - classify
    step = PrimitiveStep(primitive_description=VertexClassification.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_hyperparameter('jvm_memory', ArgumentType.VALUE, 0.6)
    step.add_output('produce')
    vertex_classification_pipeline.add_step(step)

    # Adding output step to the pipeline
    vertex_classification_pipeline.add_output(name='output', data_reference='steps.1.produce')

    return vertex_classification_pipeline