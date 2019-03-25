import sys
from typing import List, Dict, Any, Tuple, Set
import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from exline.primitives.simple_imputer import SimpleImputerPrimitive
from exline.primitives.categorical_imputer import CategoricalImputerPrimitive
from exline.primitives.standard_scaler import StandardScalerPrimitive
from exline.primitives.ensemble_forest import EnsembleForestPrimitive
from exline.primitives.replace_singletons import ReplaceSingletonsPrimitive
from exline.primitives.one_hot_encoder import OneHotEncoderPrimitive
from exline.primitives.binary_encoder import BinaryEncoderPrimitive
from exline.primitives.enrich_dates import EnrichDatesPrimitive
from exline.primitives.missing_indicator import MissingIndicatorPrimitive
from exline.primitives.simple_column_parser import SimpleColumnParserPrimitive

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.remove_columns import RemoveColumnsPrimitive

from exline.preprocessing.utils import MISSING_VALUE_INDICATOR

PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)

# CDB: Totally unoptimized.
def create_pipeline(inputs: container.DataFrame,
                    column_types: Dict[str, type],
                    target: str,
                    metric: str,
                    cat_mode: str = 'one_hot',
                    max_one_hot: int = 16,
                    scale: bool = False) -> Pipeline:

    # generate a list of the column types sorted by index
    column_indices = [(inputs.columns.get_loc(k), v) for k, v in column_types.items()]
    column_indices.sort()
    col_type_list = [t[1] for t in column_indices]

    # move d3m index to df index and drop the target - this will also be done by the column parser at pipeline
    inputs = inputs.set_index('d3mIndex')
    inputs = inputs.drop(target, axis=1)

    previous_step = 0
    input_val = 'steps.{}.produce'

    # create the basic pipeline
    tabular_pipeline = Pipeline(context=PipelineContext.TESTING)
    tabular_pipeline.add_input(name='inputs')

    # step 0 - extract dataframe from dataset
    step = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    tabular_pipeline.add_step(step)

    # step 1 - Append column parser.  D3M dataset loader creates a dataframe with all columns set to 'object', this pipeline is
    # designed to work with string/object, int, float, boolean.  This also shifts the d3mIndex to be the dataframe index and
    # drop the target.
    step = PrimitiveStep(primitive_description=SimpleColumnParserPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_output('produce_target')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # step 2 - append date enricher.  Looks for date columns and normalizes them.
    step = PrimitiveStep(primitive_description=EnrichDatesPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # step 3 - append singleton replacer.  Looks for categorical types of a single value and replaces them with a flag.
    step = PrimitiveStep(primitive_description=ReplaceSingletonsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # step 4 - Append categorical imputer.  Finds missing categorical values and replaces them with an imputed value.
    step = PrimitiveStep(primitive_description=CategoricalImputerPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # step 5 - Adds a one hot encoder for categoricals of low cardinality.
    step = PrimitiveStep(primitive_description=OneHotEncoderPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('max_one_hot', ArgumentType.VALUE, max_one_hot)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # step 6 - Adds a binary encoder for categoricals of high cardinality.
    step = PrimitiveStep(primitive_description=BinaryEncoderPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('min_binary', ArgumentType.VALUE, max_one_hot + 1)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # TODO: figure out how to integrate this - columns added?
    # step 7 - Appends a missing value transformer for numerical values.
    # step = PrimitiveStep(primitive_description=MissingIndicatorPrimitive.metadata.query())
    # step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    # step.add_output('produce')
    # tabular_pipeline.add_step(step)
    # previous_step += 1

    # step 8 - Appends an imputer for numerical values.
    step = PrimitiveStep(primitive_description=SimpleImputerPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # step 9 - Append scaler for numerical values.
    if scale:
        step = PrimitiveStep(primitive_description=StandardScalerPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        tabular_pipeline.add_step(step)
        previous_step += 1

    # step 10 - Runs a random forest ensemble.
    step = PrimitiveStep(primitive_description=EnsembleForestPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce_target')
    step.add_output('produce')
    step.add_hyperparameter('metric', ArgumentType.VALUE, metric)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    tabular_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return tabular_pipeline