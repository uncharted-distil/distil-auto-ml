import sys
from typing import List, Dict, Any, Tuple, Set
import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from distil.primitives.ragged_dataset_loader import RaggedDatasetLoaderPrimitive
from distil.primitives.timeseries_reshaper import TimeSeriesReshaperPrimitive
from distil.primitives.timeseries_neighbours import TimeSeriesNeighboursPrimitive

from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.remove_columns import RemoveColumnsPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive

PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)


def create_pipeline(metric: str) -> Pipeline:
    previous_step = 0
    input_val = 'steps.{}.produce'

    # create the basic pipeline
    ts_pipeline = Pipeline(context=PipelineContext.TESTING)
    ts_pipeline.add_input(name='inputs')

    # 0 - Extract ragged dataset to a dataframe
    step = PrimitiveStep(primitive_description=RaggedDatasetLoaderPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    step.add_output('produce_collection')
    step.add_hyperparameter('sample', ArgumentType.VALUE, 1.0)
    ts_pipeline.add_step(step)

    # 1 - Parse columns.
    step = PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_output('produce')
    semantic_types = ('http://schema.org/Boolean', 'http://schema.org/Integer', 'http://schema.org/Float',
                      'https://metadata.datadrivendiscovery.org/types/FloatVector')
    step.add_hyperparameter('parse_semantic_types', ArgumentType.VALUE, semantic_types)
    ts_pipeline.add_step(step)

    # 2 - Extract attributes
    step = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step.add_output('produce')
    step.add_hyperparameter('semantic_types', ArgumentType.VALUE, ('https://metadata.datadrivendiscovery.org/types/Attribute',))
    ts_pipeline.add_step(step)

    # 3- Extract targets
    step = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step.add_output('produce')
    target_types = ('https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
    step.add_hyperparameter('semantic_types', ArgumentType.VALUE, target_types)
    ts_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    # 4 - Reformats timeseries data if sparse, truncates if required
    step = PrimitiveStep(primitive_description=TimeSeriesReshaperPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce_collection')
    step.add_output('produce')
    ts_pipeline.add_step(step)

    # 5 - Classification / regression of series data
    step = PrimitiveStep(primitive_description=TimeSeriesNeighboursPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step.add_output('produce')
    step.add_hyperparameter('metric', ArgumentType.VALUE, metric)
    ts_pipeline.add_step(step)

    # step 4 - convert predictions to expected format
    step = PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
    step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step.add_output('produce')
    ts_pipeline.add_step(step)

    # Adding output step to the pipeline
    ts_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return ts_pipeline
