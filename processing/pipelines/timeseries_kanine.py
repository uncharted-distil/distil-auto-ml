import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from d3m.primitives.time_series_classification.k_neighbors import Kanine
from distil.primitives.timeseries_formatter import TimeSeriesFormatterPrimitive

def create_pipeline(metric: str) -> Pipeline:

    # create the basic pipeline
    kanine_pipeline = Pipeline()
    kanine_pipeline.add_input(name='inputs')

    # step 0 - flatten the timeseries if necessary
    step = PrimitiveStep(primitive_description=TimeSeriesFormatterPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    kanine_pipeline.add_step(step)

    # step 1 - kanine classification
    step = PrimitiveStep(primitive_description=Kanine.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_hyperparameter('long_format', ArgumentType.VALUE, True)
    step.add_output('produce')
    kanine_pipeline.add_step(step)

    # Adding output step to the pipeline
    kanine_pipeline.add_output(name='output', data_reference='steps.1.produce')

    return kanine_pipeline