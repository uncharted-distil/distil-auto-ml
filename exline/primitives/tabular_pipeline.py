import sys
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from d3m import container
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType

from primitives.simple_imputer import SimpleImputerPrimitive
from primitives.categorical_imputer import CategoricalImputerPrimitive
from primitives.standard_scaler import StandardScalerPrimitive
from primitives.ensemble_forest import EnsembleForestPrimitive
from sklearn_wrap import SKOneHotEncoder, SKMissingIndicator

from preprocessing.utils import MISSING_VALUE_INDICATOR


def create_pipeline(inputs: container.DataFrame,
                    target_idx: int,
                    metric: str,
                    cat_mode: str = 'one_hot',
                    max_one_hot: int = 16,
                    scale: bool = False) -> Pipeline:

    previous_step = 0
    input_val = 'steps.${s}.produce'

    tabular_pipeline = Pipeline()
    tabular_pipeline.add_input(name='inputs')

    # append date enricher
    step = PrimitiveStep(primitive_description=SimpleImputerPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # append single replacer
    step = PrimitiveStep(primitive_description=SimpleImputerPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # map of operations to apply to each column
    categorical_imputer_cols: List[Tuple[int, List[Any]]] = []
    svm_text_cols: List[int] = []
    one_hot_cols: List[int] = []
    binary_cols: List[int] = []

    simple_imputer_cols: List[int] = []
    standard_scalar_cols: List[int] = []
    missing_indicator_cols: List[int] = []

    # loop over dataset columns and append column specific transformations
    for i, c in enumerate(inputs.columns):
        if inputs[c].dtype == np.object_:
            uvals = list(set(inputs[c]))
            if len(uvals) == 1:
                print('%s has 1 level -> skipping' % c, file=sys.stderr)
                continue

            categories = [uvals + [MISSING_VALUE_INDICATOR]]

            # map encoders to column number
            categorical_imputer_cols.append((i, categories))
            if detect_text(inputs[c]):
                svm_text_cols.append(i)
            elif (cat_mode == 'one_hot') and (len(uvals) < max_one_hot):
                one_hot_cols.append(i)
            else:
                binary_cols.append(i)

        elif inputs[c].dtype in [float, int, bool]:
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
        step.add_hyperparameter('use_columns', List[int], [col[0] for col in categorical_imputer_cols])
        tabular_pipeline.add_step(step)
        previous_step += 1

    # append one hot encoder if necessary
    if len(one_hot_cols) > 0:
        step = PrimitiveStep(primitive_description=SKOneHotEncoder.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', List[int], [col[0] for col in categorical_imputer_cols])
        step.add_hyperparameter('sparse', bool, False)
        step.add_hyperparameter('handle_unknown', str, 'ignore')
        step.add_hyperparameter('categories', [c[1] for c in categorical_imputer_cols])
        tabular_pipeline.add_step(step)
        previous_step += 1

    # skip text, binary encoders for now
    if len(svm_text_cols) > 0 or len(binary_cols) > 0:
        raise NotImplemented

    # append simple imputer if necessary
    if len(simple_imputer_cols) > 0:
        step = PrimitiveStep(primitive_description=SimpleImputerPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', List[int], simple_imputer_cols)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # append simple imputer if necessary
    if len(standard_scalar_cols) > 0:
        step = PrimitiveStep(primitive_description=StandardScalerPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', List[int], standard_scalar_cols)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # append missing indicator if necessary
    if len(missing_indicator_cols) > 0:
        step = PrimitiveStep(primitive_description=SKMissingIndicator.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('use_columns', List[int], standard_scalar_cols)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # run a random forest ensemble
    step = PrimitiveStep(primitive_description=EnsembleForestPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('target_index', int, target_idx)
    step.add_hyperparameter('metric', str, metric)
    tabular_pipeline.add_step(step)

    return tabular_pipeline


def detect_text(X: container.ndarray, thresh: int = 8) -> container.ndarray:
    """ returns true if median entry has more than `thresh` tokens"""
    X = X[X.notnull()]
    n_toks = X.apply(lambda xx: len(str(xx).split(' '))).values
    return np.median(n_toks) >= thresh