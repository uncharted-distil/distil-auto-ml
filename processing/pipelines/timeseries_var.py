import sys
from typing import List, Dict, Any, Tuple, Set, Optional
import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from d3m.primitives.time_series_forecasting.vector_autoregression import VAR
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.column_parser import ColumnParserPrimitive


def create_pipeline(metric: str, resolver: Optional[Resolver] = None) -> Pipeline:
    previous_step = 0
    input_val = 'steps.{}.produce'

    # create the basic pipeline
    var_pipeline = Pipeline()
    var_pipeline.add_input(name='inputs')

    # step 0 - Extract dataframe from dataset
    step = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    var_pipeline.add_step(step)

    # step 1 - Parse columns.
    step = PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    semantic_types = ('http://schema.org/Boolean', 'http://schema.org/Integer', 'http://schema.org/Float',
                      'https://metadata.datadrivendiscovery.org/types/FloatVector')
    step.add_hyperparameter('parse_semantic_types', ArgumentType.VALUE, semantic_types)
    var_pipeline.add_step(step)
    previous_step += 1
    parse_step = previous_step

    # step 2 - Vector Auto Regression
    step = PrimitiveStep(primitive_description=VAR.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step.add_output('produce')
    var_pipeline.add_step(step)
    previous_step += 1
    # Adding output step to the pipeline
    var_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return var_pipeline