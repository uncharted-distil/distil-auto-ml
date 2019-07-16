import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from d3m.primitives.time_series_classification.convolutional_neural_net import LSTM_FCN
from common_primitives.denormalize import DenormalizePrimitive

PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)


def create_pipeline(metric: str) -> Pipeline:
    
    # create the basic pipeline
    lstm_fcn_pipeline = Pipeline(context=PipelineContext.TESTING)
    lstm_fcn_pipeline.add_input(name='inputs')

    # Denormalize so that we have a single dataframe in the dataset
    step = PrimitiveStep(primitive_description=DenormalizePrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    lstm_fcn_pipeline.add_step(step)

    # step 1 - LSTM FCN classification
    step = PrimitiveStep(primitive_description=LSTM_FCN.metadata.query()) 
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_hyperparameter(name='attention_lstm', argument_type= ArgumentType.VALUE, data=True)
    step.add_hyperparameter(name='lstm_cells', argument_type= ArgumentType.VALUE, data=64)
    step.add_output('produce')
    lstm_fcn_pipeline.add_step(step)

    # Adding output step to the pipeline
    lstm_fcn_pipeline.add_output(name='output', data_reference='steps.1.produce')

    return lstm_fcn_pipeline