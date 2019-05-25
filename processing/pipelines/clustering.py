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
from exline.primitives.replace_singletons import ReplaceSingletonsPrimitive
from exline.primitives.one_hot_encoder import OneHotEncoderPrimitive
from exline.primitives.binary_encoder import BinaryEncoderPrimitive
from exline.primitives.text_encoder import TextEncoderPrimitive
from exline.primitives.enrich_dates import EnrichDatesPrimitive
from exline.primitives.missing_indicator import MissingIndicatorPrimitive
from exline.primitives.simple_column_parser import SimpleColumnParserPrimitive
from exline.primitives.zero_column_remover import ZeroColumnRemoverPrimitive
from exline.primitives.k_means import KMeansPrimitive

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.remove_columns import RemoveColumnsPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive

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
                    scale: bool = False,
                    num_clusters: int = 100) -> Pipeline:
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

    # Append column parser.  D3M dataset loader creates a dataframe with all columns set to 'object', this pipeline is
    # designed to work with string/object, int, float, boolean.  This also shifts the d3mIndex to be the dataframe index and
    # drop the target.
    step = PrimitiveStep(primitive_description=SimpleColumnParserPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_output('produce_target')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Append date enricher.  Looks for date columns and normalizes them.
    step = PrimitiveStep(primitive_description=EnrichDatesPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Append singleton replacer.  Looks for categorical values that only occur once in a column and replace them with a flag.
    step = PrimitiveStep(primitive_description=ReplaceSingletonsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Append categorical imputer.  Finds missing categorical values and replaces them with an imputed value.
    step = PrimitiveStep(primitive_description=CategoricalImputerPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Adds a one hot encoder for categoricals of low cardinality.
    step = PrimitiveStep(primitive_description=OneHotEncoderPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('max_one_hot', ArgumentType.VALUE, max_one_hot)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Adds a binary encoder for categoricals of high cardinality.
    step = PrimitiveStep(primitive_description=BinaryEncoderPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('min_binary', ArgumentType.VALUE, max_one_hot + 1)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Appends a missing value transformer for numerical values.
    step = PrimitiveStep(primitive_description=MissingIndicatorPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Appends an imputer for numerical values.
    step = PrimitiveStep(primitive_description=SimpleImputerPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Append scaler for numerical values.
    if scale:
        step = PrimitiveStep(primitive_description=StandardScalerPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        tabular_pipeline.add_step(step)
        previous_step += 1

    # Adds a k-mean clustering step
    step = PrimitiveStep(primitive_description=KMeansPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('n_clusters', ArgumentType.VALUE, num_clusters)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # convert predictions to expected format
    step = PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_output('produce')
    step.add_hyperparameter('use_columns', ArgumentType.VALUE, [0, 1])
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    tabular_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return tabular_pipeline