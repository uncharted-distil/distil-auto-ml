import sys
from typing import List, Dict, Any, Tuple, Set
import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from distil.primitives.simple_column_parser import SimpleColumnParserPrimitive
from distil.primitives.bert_classification import BertClassificationPrimitive

from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.remove_columns import RemoveColumnsPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive

PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)


def create_pipeline(metric: str) -> Pipeline:
    previous_step = 0
    input_val = 'steps.{}.produce'

    # create the basic pipeline
    qa_pipeline = Pipeline(context=PipelineContext.TESTING)
    qa_pipeline.add_input(name='inputs')

    # Denormalize so that we have a single dataframe in the dataset
    step = PrimitiveStep(primitive_description=DenormalizePrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    qa_pipeline.add_step(step)

    # Extract dataframe from dataset
    step = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    qa_pipeline.add_step(step)
    previous_step += 1

    # Append column parser.  D3M dataset loader creates a dataframe with all columns set to 'object', this pipeline is
    # designed to work with string/object, int, float, boolean.  This also shifts the d3mIndex to be the dataframe index and
    # drop the target.
    step = PrimitiveStep(primitive_description=SimpleColumnParserPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_output('produce_target')
    qa_pipeline.add_step(step)
    previous_step += 1

    # Generates a bert pair classification model.
    step = PrimitiveStep(primitive_description=BertClassificationPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce_target')
    step.add_output('produce')
    step.add_hyperparameter('metric', ArgumentType.VALUE, metric)
    # step.add_hyperparameter('fast', ArgumentType.VALUE, True)
    qa_pipeline.add_step(step)
    previous_step += 1

    # convert predictions to expected format
    step = PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce_target')
    step.add_output('produce')
    step.add_hyperparameter('use_columns', ArgumentType.VALUE, [0, 1])
    qa_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    qa_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return qa_pipeline
