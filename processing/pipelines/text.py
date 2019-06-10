import sys
from typing import List, Dict, Any, Tuple, Set
import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams


from distil.primitives.text_classifier import TextClassifierPrimitive

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive

from common_primitives.construct_predictions import ConstructPredictionsPrimitive

from distil.primitives.simple_column_parser import SimpleColumnParserPrimitive

from common_primitives.denormalize import DenormalizePrimitive

from common_primitives.text_reader import TextReaderPrimitive


PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)

# CDB: Totally unoptimized.  Pipeline creation code could be simplified but has been left
# in a naively implemented state for readability for now.
#
# Overall implementation relies on passing the entire dataset through the pipeline, with the primitives
# identifying columns to operate on based on type.  Alternative implementation (that better lines up with
# D3M approach, but generates more complex pipelines) would be to extract sub-sets by semantic type using
# a common primitive, apply the type-specific primitive to the sub-set, and then merge the changes
# (replace or join) back into the original data.
def create_pipeline(metric: str,
                    cat_mode: str = 'one_hot',
                    max_one_hot: int = 16,
                    scale: bool = False) -> Pipeline:
    previous_step = 0
    input_val = 'steps.{}.produce'

    # create the basic pipeline
    text_pipeline = Pipeline(context=PipelineContext.TESTING)
    text_pipeline.add_input(name='inputs')

    # step 0 - denormalize dataframe (injects semantic type information)
    step = PrimitiveStep(primitive_description=DenormalizePrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    text_pipeline.add_step(step)

    # step 1 - extract dataframe from dataset
    step = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_output('produce')
    text_pipeline.add_step(step)

    # step 2 - read text
    step = PrimitiveStep(primitive_description=TextReaderPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step.add_output('produce')
    step.add_hyperparameter('use_columns', ArgumentType.VALUE,[0,1])
    step.add_hyperparameter('return_result', ArgumentType.VALUE, 'replace')
    text_pipeline.add_step(step)

    # step 3 - set up training set
    step = PrimitiveStep(primitive_description=SimpleColumnParserPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step.add_output('produce')
    step.add_output('produce_target')
    text_pipeline.add_step(step)


    # step 4 - generates a text classification model.
    step = PrimitiveStep(primitive_description=TextClassifierPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce_target')
    step.add_output('produce')
    step.add_hyperparameter('metric', ArgumentType.VALUE, metric)
    step.add_hyperparameter('fast', ArgumentType.VALUE, True)
    text_pipeline.add_step(step)


    # step 5 - convert predictions to expected format
    step = PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce_target')
    step.add_output('produce')
    step.add_hyperparameter('use_columns', ArgumentType.VALUE, [0, 1])
    text_pipeline.add_step(step)


    # Adding output step to the pipeline
    text_pipeline.add_output(name='output', data_reference='steps.5.produce')

    return text_pipeline