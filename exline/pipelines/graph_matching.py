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
from exline.primitives.text_encoder import TextEncoderPrimitive
from exline.primitives.enrich_dates import EnrichDatesPrimitive
from exline.primitives.missing_indicator import MissingIndicatorPrimitive
from exline.primitives.simple_column_parser import SimpleColumnParserPrimitive
from exline.primitives.zero_column_remover import ZeroColumnRemoverPrimitive

from exline.primitives.load_graphs import LoadGraphsPrimitive

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.remove_columns import RemoveColumnsPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive

from exline.preprocessing.utils import MISSING_VALUE_INDICATOR

PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)


'''
        graphs = load_graphs(d3mds)
        assert len(graphs) == 2
        
        sgm = SGMGraphMatcher(target_metric=ll_metric)
        test_score = sgm.fit_score(graphs, X_train, X_test, y_train, y_test)
        
        _extra = {
            "train_acc"      : sgm.train_acc,
            "null_train_acc" : sgm.null_train_acc,
            
            "test_acc"       : sgm.test_acc,
            "null_test_acc"  : sgm.null_test_acc,
}
'''


def create_pipeline(metric: str,
                    cat_mode: str = 'one_hot',
                    max_one_hot: int = 16,
                    scale: bool = False) -> Pipeline:
    previous_step = 0
    input_val = 'steps.{}.produce'

    # create the basic pipeline
    graph_matching_pipeline = Pipeline(context=PipelineContext.TESTING)
    graph_matching_pipeline.add_input(name='inputs')

    # step 0 - load the graphs 
    step = PrimitiveStep(primitive_description=LoadGraphsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    graph_matching_pipeline.add_step(step)

    
    # convert predictions to expected format
    step = PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce_target')
    step.add_output('produce')
    step.add_hyperparameter('use_columns', ArgumentType.VALUE, [0, 1])
    graph_matching_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    graph_matching_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return graph_matching_pipeline