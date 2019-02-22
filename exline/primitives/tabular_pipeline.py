import sys
from typing import List, Dict, Any, Tuple, Set

import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from primitives.simple_imputer import SimpleImputerPrimitive
from primitives.categorical_imputer import CategoricalImputerPrimitive
from primitives.standard_scaler import StandardScalerPrimitive
from primitives.ensemble_forest import EnsembleForestPrimitive
from primitives.replace_singletons import ReplaceSingletonsPrimitive
from primitives.one_hot_encoder import OneHotEncoderPrimitive
from primitives.binary_encoder import BinaryEncoderPrimitive
from primitives.enrich_dates import EnrichDatesPrimitive
from sklearn_wrap.SKMissingIndicator import SKMissingIndicator
from common_primitives.column_parser import ColumnParserPrimitive

from preprocessing.utils import MISSING_VALUE_INDICATOR

PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)

# CDB: Totally unoptimized.
def create_pipeline(inputs: container.DataFrame,
                    column_types: Dict[int, str],
                    target_idx: int,
                    metric: str,
                    cat_mode: str = 'one_hot',
                    max_one_hot: int = 16,
                    scale: bool = False) -> Pipeline:

    previous_step = 0
    input_val = 'steps.{}.produce'

    # create the basic pipeline
    tabular_pipeline = Pipeline(context=PipelineContext.TESTING)
    tabular_pipeline.add_input(name='inputs')

    # append column parser
    step = PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    tabular_pipeline.add_step(step)

    # append date enricher
    step = PrimitiveStep(primitive_description=EnrichDatesPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)

    # append singleton replacer
    step = PrimitiveStep(primitive_description=ReplaceSingletonsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # map of operations to apply to each column
    categorical_imputer_cols: Dict[int, Set[Any]] = {}
    svm_text_cols: List[int] = []
    one_hot_cols: List[int] = []
    binary_cols: List[int] = []

    simple_imputer_cols: List[int] = []
    standard_scalar_cols: List[int] = []
    missing_indicator_cols: List[int] = []

    # loop over dataset columns and determine which primitives are needed, and which columns
    # they should each be applied to
    for i, c in enumerate(inputs.columns):
        if column_types[i] is str:
            uvals = set(inputs[c])
            if len(uvals) == 1:
                print('%s has 1 level -> skipping' % c, file=sys.stderr)
                continue

            uvals.add(MISSING_VALUE_INDICATOR)

            # map encoders to column number
            categorical_imputer_cols[i] = uvals
            if detect_text(inputs[c]):
                svm_text_cols.append(i)
            elif (cat_mode == 'one_hot') and (len(uvals) < max_one_hot):
                one_hot_cols.append(i)
            else:
                binary_cols.append(i)

        elif column_types[i] in [float, int, bool]:
            simple_imputer_cols.append(i)
            if scale:
                raise Exception
                standard_scalar_cols.append(i)

            if inputs[c].isnull().any():
                missing_indicator_cols.append(i)
        else:
            raise NotImplemented

    # append categorical imputer
    if len(categorical_imputer_cols) > 0:
        step = PrimitiveStep(primitive_description=CategoricalImputerPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', ArgumentType.VALUE, [c for c in categorical_imputer_cols])
        tabular_pipeline.add_step(step)
        previous_step += 1

    # append one hot encoder for categoricals of low cardinality
    if len(one_hot_cols) > 0:
        step = PrimitiveStep(primitive_description=OneHotEncoderPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', ArgumentType.VALUE, [c for c in one_hot_cols])
        step.add_hyperparameter('categories', ArgumentType.VALUE, [categorical_imputer_cols[c] for c in one_hot_cols])
        tabular_pipeline.add_step(step)
        previous_step += 1

    # skip text encoders for now
    if len(svm_text_cols) > 0:
        raise NotImplemented

    # append a binary encoder for categoricals of high cardinality
    if len(binary_cols) > 0:
        step = PrimitiveStep(primitive_description=BinaryEncoderPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', ArgumentType.VALUE, binary_cols)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # append simple imputer for numerics
    if len(simple_imputer_cols) > 0:
        step = PrimitiveStep(primitive_description=SimpleImputerPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', ArgumentType.VALUE, simple_imputer_cols)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # append the standard scalar for numerics
    if len(standard_scalar_cols) > 0:
        step = PrimitiveStep(primitive_description=StandardScalerPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', ArgumentType.VALUE, standard_scalar_cols)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # append missing indicator if necessary
    if len(missing_indicator_cols) > 0:
        step = PrimitiveStep(primitive_description=SKMissingIndicator.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', ArgumentType.VALUE, missing_indicator_cols)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # run a random forest ensemble
    step = PrimitiveStep(primitive_description=EnsembleForestPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('target_idx', ArgumentType.VALUE, target_idx)
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