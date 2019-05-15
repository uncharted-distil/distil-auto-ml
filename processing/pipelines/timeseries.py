import sys
from typing import List, Dict, Any, Tuple, Set
import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from exline.primitives.simple_column_parser import SimpleColumnParserPrimitive
from exline.primitives.ragged_dataset_loader import RaggedDatasetLoaderPrimitive
from exline.primitives.timeseries_reshaper import TimeSeriesReshaperPrimitive
from exline.primitives.timeseries_neighbours import TimeSeriesNeighboursPrimitive

from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.remove_columns import RemoveColumnsPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive

PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)


def create_pipeline(metric: str) -> Pipeline:
    previous_step = 0
    input_val = 'steps.{}.produce'

    # create the basic pipeline
    ts_pipeline = Pipeline(context=PipelineContext.TESTING)
    ts_pipeline.add_input(name='inputs')

    # step 0 - extract ragged dataset to a dataframe
    step = PrimitiveStep(primitive_description=RaggedDatasetLoaderPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    step.add_output('produce_collection')
    step.add_hyperparameter('sample', ArgumentType.VALUE, 1.0)
    ts_pipeline.add_step(step)

    # step 1 - Append column parser.  D3M dataset loader creates a dataframe with all columns set to 'object', this pipeline is
    # designed to work with string/object, int, float, boolean.  This also shifts the d3mIndex to be the dataframe index and
    # drop the target.
    step = PrimitiveStep(primitive_description=SimpleColumnParserPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_output('produce')
    step.add_output('produce_target')
    ts_pipeline.add_step(step)

    # step 2 - Reformats timeseries data if sparse, truncates if required
    step = PrimitiveStep(primitive_description=TimeSeriesReshaperPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce_collection')
    step.add_output('produce')
    ts_pipeline.add_step(step)

    # step 3 - Classification / regression of series data
    step = PrimitiveStep(primitive_description=TimeSeriesNeighboursPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce_target')
    step.add_output('produce')
    step.add_hyperparameter('metric', ArgumentType.VALUE, metric)
    ts_pipeline.add_step(step)

    # step 4 - convert predictions to expected format
    step = PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce_target')
    step.add_output('produce')
    ts_pipeline.add_step(step)

    # Adding output step to the pipeline
    ts_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return ts_pipeline
