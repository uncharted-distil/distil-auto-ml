import sys
from typing import List, Dict, Any, Tuple, Set
import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from exline.primitives.load_graphs import ExlineGraphLoaderPrimitive
from exline.primitives.seeded_graph_matcher import ExlineSeededGraphMatchingPrimitive

from exline.primitives.simple_column_parser import SimpleColumnParserPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive


from common_primitives.construct_predictions import ConstructPredictionsPrimitive

from exline.preprocessing.utils import MISSING_VALUE_INDICATOR

PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)



def create_pipeline(metric: str) -> Pipeline:

    # create the basic pipeline
    graph_matching_pipeline = Pipeline(context=PipelineContext.TESTING)
    graph_matching_pipeline.add_input(name='inputs')

    # step 0 - extract the graphs 
    step = PrimitiveStep(primitive_description=ExlineGraphLoaderPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    step.add_output('produce_target')
    graph_matching_pipeline.add_step(step)

    # step 1 - match the graphs that have been seeded 
    step = PrimitiveStep(primitive_description=ExlineSeededGraphMatchingPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce_target')
    step.add_hyperparameter('metric', ArgumentType.VALUE, metric)
    step.add_output('produce')
    graph_matching_pipeline.add_step(step)

    # convert predictions to expected format
    #step = PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
    #step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    #step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce_target')
    #step.add_output('produce')
    #step.add_hyperparameter('use_columns', ArgumentType.VALUE, [0, 1])
    #graph_matching_pipeline.add_step(step)

    # Adding output step to the pipeline
    graph_matching_pipeline.add_output(name='output', data_reference='steps.1.produce')

    return graph_matching_pipeline