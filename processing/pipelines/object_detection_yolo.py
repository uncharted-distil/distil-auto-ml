import sys
from typing import List, Dict, Any, Tuple, Set,Optional
import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.dataframe_image_reader import DataFrameImageReaderPrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
from common_primitives.column_parser import ColumnParserPrimitive



from dsbox.datapreprocessing.featurizer.image.object_detection import Yolo

from distil.primitives.ensemble_forest import EnsembleForestPrimitive
from distil.primitives.image_transfer import ImageTransferPrimitive

def create_pipeline(metric: str,
                    cat_mode: str = 'one_hot',
                    max_one_hot: int = 16,
                    scale: bool = False,
                    resolver: Optional[Resolver] = None) -> Pipeline:

    # create the basic pipeline
    objdetect_pipeline = Pipeline()
    objdetect_pipeline.add_input(name='inputs')

    # step 0 - denormalize dataframe (N.B.: injects semantic type information)
    step = PrimitiveStep(primitive_description=DenormalizePrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    step.add_hyperparameter('starting_resource', ArgumentType.VALUE, None)
    step.add_hyperparameter('recursive', ArgumentType.VALUE, True)
    step.add_hyperparameter('many_to_many', ArgumentType.VALUE, False)
    step.add_hyperparameter('discard_not_joined_tabular_resources', ArgumentType.VALUE, False)
    objdetect_pipeline.add_step(step)

    # step 1 - extract dataframe from dataset
    step = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_output('produce')
    objdetect_pipeline.add_step(step)

    # step 2 - extract files
    step = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step.add_output('produce')
    target_types = ('https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey', 'https://metadata.datadrivendiscovery.org/types/FileName')
    step.add_hyperparameter('semantic_types', ArgumentType.VALUE, target_types)
    objdetect_pipeline.add_step(step)

    # step 3 - extract targets
    step = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step.add_output('produce')
    target_types = ('https://metadata.datadrivendiscovery.org/types/TrueTarget',)
    step.add_hyperparameter('semantic_types', ArgumentType.VALUE, target_types)
    objdetect_pipeline.add_step(step)

    # step 4 - extract objects
    step = PrimitiveStep(primitive_description=Yolo.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step.add_output('produce')
    objdetect_pipeline.add_step(step)

    # Adding output step to the pipeline
    objdetect_pipeline.add_output(name='output', data_reference='steps.4.produce')

    return objdetect_pipeline