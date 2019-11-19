import sys
from typing import List, Dict, Any, Tuple, Set
import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

# from d3m.primitives.time_series_forecasting.vector_autoregression import VAR
from d3m.primitives.time_series_forecasting.arima import Parrot

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.column_parser import ColumnParserPrimitive

PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)


def create_pipeline(metric: str) -> Pipeline:
    previous_step = 0
    input_val = 'steps.{}.produce'

    # create the basic pipeline
    tsf_pipeline = Pipeline(context=PipelineContext.TESTING)
    tsf_pipeline.add_input(name='inputs')

    # step 0 - Extract dataframe from dataset
    step = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    tsf_pipeline.add_step(step)

    # step 1 - Parse columns.
    step = PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_output('produce')
    semantic_types = ('http://schema.org/Boolean', 'http://schema.org/Integer', 'http://schema.org/Float',
                      'https://metadata.datadrivendiscovery.org/types/FloatVector')
    step.add_hyperparameter('parse_semantic_types', ArgumentType.VALUE, semantic_types)
    tsf_pipeline.add_step(step)



    # step 2 - Parrot ARIMA
    step = PrimitiveStep(primitive_description=Parrot.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step.add_hyperparameter(name='seasonal_differencing', argument_type= ArgumentType.VALUE, data=11)
    step.add_hyperparameter(name='n_periods', argument_type= ArgumentType.VALUE, data=21)
    step.add_output('produce')
    tsf_pipeline.add_step(step)

    # step 3 - convert predictions to expected format
    # step = PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
    # step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    # step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    # step.add_output('produce')
    # tsf_pipeline.add_step(step)

    # Adding output step to the pipeline
    tsf_pipeline.add_output(name='output', data_reference='steps.2.produce')

    return tsf_pipeline