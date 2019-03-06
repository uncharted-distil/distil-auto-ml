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

#import d3m.primitives.data_transformation.imputer.ExlineSimpleImputer as SimpleImputerPrimitive
#import d3m.primitives.data_transformation.imputer.ExlineCategoricalImputer as CategoricalImputerPrimitive

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
    print(inputs.columns)
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
    # designed to work with string/object, int, float, boolean.  This also shifts the d3mIndex to be the dataframe index.
    step = PrimitiveStep(primitive_description=SimpleColumnParserPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_output('produce_target')
    step.add_hyperparameter('column_types', ArgumentType.VALUE, col_type_list)
    step.add_hyperparameter('target', ArgumentType.VALUE, target)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # step 2 - append date enricher
    step = PrimitiveStep(primitive_description=EnrichDatesPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # step 3 - append singleton replacer
    step = PrimitiveStep(primitive_description=ReplaceSingletonsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # map of operations to apply to each column
    categorical_imputer_cols: List[int] = []
    svm_text_cols: List[int] = []
    one_hot_cols: List[int] = []
    binary_cols: List[int] = []

    simple_imputer_cols: List[int] = []
    standard_scalar_cols: List[int] = []
    missing_indicator_cols: List[int] = []

    # loop over dataset columns and determine which primitives are needed, and which columns
    # they should each be applied to
    for c in inputs.columns:
        col_idx = inputs.columns.get_loc(c)
        if column_types[c] is str:
            uvals = set(inputs[c])
            if len(uvals) == 1:
                print('%s has 1 level -> skipping' % c, file=sys.stderr)
                continue


            # map encoders to column number
            categorical_imputer_cols.append(col_idx)
            if detect_text(inputs[c]):
                svm_text_cols.append(col_idx)
            elif (cat_mode == 'one_hot') and (len(uvals) + 1 < max_one_hot): # add 1 for MISSING_VALUE_INDICATOR
                print(c)
                one_hot_cols.append(col_idx)
            else:
                binary_cols.append(col_idx)

        elif column_types[c] in [float, int, bool]:
            simple_imputer_cols.append(col_idx)
            if scale:
                raise Exception
                standard_scalar_cols.append(col_idx)

            if inputs[c].isnull().any():
                missing_indicator_cols.append(col_idx)
        else:
            raise NotImplemented

    print(inputs)

    # step 4 - append categorical imputer
    if len(categorical_imputer_cols) > 0:
        step = PrimitiveStep(primitive_description=CategoricalImputerPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', ArgumentType.VALUE, [c for c in categorical_imputer_cols])
        tabular_pipeline.add_step(step)
        previous_step += 1

    # step 5 - a append one hot encoder for categoricals of low cardinality
    if len(one_hot_cols) > 0:
        step = PrimitiveStep(primitive_description=OneHotEncoderPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', ArgumentType.VALUE, [c for c in one_hot_cols])
        tabular_pipeline.add_step(step)
        previous_step += 1


    # skip text encoders for now
    if len(svm_text_cols) > 0:
        raise NotImplemented

    # step 6 - append a binary encoder for categoricals of high cardinality
    if len(binary_cols) > 0:
        step = PrimitiveStep(primitive_description=BinaryEncoderPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', ArgumentType.VALUE, binary_cols)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # step 7 - append missing indicator if necessary
    # if len(missing_indicator_cols) > 0:
    #     step = PrimitiveStep(primitive_description=MissingIndicatorPrimitive.metadata.query())
    #     step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    #     step.add_output('produce')
    #     step.add_hyperparameter('use_columns', ArgumentType.VALUE, missing_indicator_cols)
    #     tabular_pipeline.add_step(step)
    #     previous_step += 1

    # # step 8 - append simple imputer for numerics
    if len(simple_imputer_cols) > 0:
        step = PrimitiveStep(primitive_description=SimpleImputerPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', ArgumentType.VALUE, simple_imputer_cols)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # step 9 - append the standard scalar for numerics
    if len(standard_scalar_cols) > 0:
        step = PrimitiveStep(primitive_description=StandardScalerPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', ArgumentType.VALUE, standard_scalar_cols)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # step 10 - drop columns that were encoded
    step = PrimitiveStep(primitive_description=RemoveColumnsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('columns', ArgumentType.VALUE, one_hot_cols + binary_cols + svm_text_cols + missing_indicator_cols)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # step 11 - run a random forest ensemble
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


def detect_text(X: container.ndarray, thresh: int = 8) -> container.ndarray:
    """ returns true if median entry has more than `thresh` tokens"""
    X = X[X.notnull()]
    n_toks = X.apply(lambda xx: len(str(xx).split(' '))).values
    return np.median(n_toks) >= thresh