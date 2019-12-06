import logging
import numpy as np
import pandas as pd
from typing import Optional

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from d3m.primitives.time_series_classification.convolutional_neural_net import LSTM_FCN
from distil.primitives.timeseries_formatter import TimeSeriesFormatterPrimitive

def create_pipeline(metric: str, resolver: Optional[Resolver] = None) -> Pipeline:

    # create the basic pipeline
    lstm_fcn_pipeline = Pipeline()
    lstm_fcn_pipeline.add_input(name='inputs')

    # step 0 - flatten the timeseries if necessary
    step = PrimitiveStep(primitive_description=TimeSeriesFormatterPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    lstm_fcn_pipeline.add_step(step)

    # step 1 - LSTM FCN classification
    step = PrimitiveStep(primitive_description=LSTM_FCN.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_hyperparameter(name='attention_lstm', argument_type= ArgumentType.VALUE, data=False)
    step.add_hyperparameter(name='lstm_cells', argument_type= ArgumentType.VALUE, data=64)
    step.add_hyperparameter('long_format', ArgumentType.VALUE, True)
    step.add_output('produce')
    lstm_fcn_pipeline.add_step(step)

    # Adding output step to the pipeline
    lstm_fcn_pipeline.add_output(name='output', data_reference='steps.1.produce')

    return lstm_fcn_pipeline