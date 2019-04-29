import sys
from typing import List, Dict, Any, Tuple, Set
import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from d3m.primitives.time_series_forecasting.vector_autoregression import VAR

from exline.primitives.simple_column_parser import SimpleColumnParserPrimitive
from exline.primitives.collaborative_filtering import CollaborativeFilteringPrimitive

from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.remove_columns import RemoveColumnsPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive

PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)


def create_pipeline(metric: str) -> Pipeline:
    previous_step = 0
    input_val = 'steps.{}.produce'

    # create the basic pipeline
    cf_pipeline = Pipeline(context=PipelineContext.TESTING)
    cf_pipeline.add_input(name='inputs')

    # Extract dataframe from dataset
    step = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    cf_pipeline.add_step(step)

    # Append column parser.  D3M dataset loader creates a dataframe with all columns set to 'object', this pipeline is
    # designed to work with string/object, int, float, boolean.  This also shifts the d3mIndex to be the dataframe index.
    # Targets are left in place.
    step = PrimitiveStep(primitive_description=SimpleColumnParserPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('include_targets', ArgumentType.VALUE, True)
    cf_pipeline.add_step(step)
    previous_step += 1

    # Vector Auto Regression for forecasting a time series - it uses the `SuggestedTarget` semantic type to determine
    # which columns to run the regression on .
    step = PrimitiveStep(primitive_description=VAR.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    cf_pipeline.add_step(step)
    previous_step += 1

    # convert predictions to expected format
    step = PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce_target')
    step.add_output('produce')
    step.add_hyperparameter('use_columns', ArgumentType.VALUE, [0, 1])
    cf_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    cf_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return cf_pipeline