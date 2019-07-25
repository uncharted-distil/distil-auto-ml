import os
import sys
from typing import List, Dict, Any, Tuple, Set, Optional
import logging
import numpy as np
import pandas as pd

import json
import requests

import config

from d3m import container, utils
from d3m.base import utils as base_utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from d3m.metadata import hyperparams

import datamart
import datamart_nyu

from distil.primitives.categorical_imputer import CategoricalImputerPrimitive
from distil.primitives.ensemble_forest import EnsembleForestPrimitive
from distil.primitives.replace_singletons import ReplaceSingletonsPrimitive
from distil.primitives.one_hot_encoder import OneHotEncoderPrimitive
from distil.primitives.binary_encoder import BinaryEncoderPrimitive
from distil.primitives.text_encoder import TextEncoderPrimitive
from distil.primitives.enrich_dates import EnrichDatesPrimitive

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.remove_columns import RemoveColumnsPrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
from common_primitives.dataset_sample import DatasetSamplePrimitive
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.replace_semantic_types import ReplaceSemanticTypesPrimitive
from common_primitives.datamart_download import DataMartDownloadPrimitive
from common_primitives.datamart_augment import DataMartAugmentPrimitive
from common_primitives.extract_columns_structural_types import ExtractColumnsByStructuralTypesPrimitive

from d3m.primitives.data_cleaning.column_type_profiler import Simon

from sklearn_wrap import SKMissingIndicator
from sklearn_wrap import SKImputer
from sklearn_wrap import SKStandardScaler
from sklearn_wrap import SKPCA

logger = logging.getLogger(__name__)

def create_pipeline(metric: str,
                    keywords: list = [],
                    dataset: container.Dataset = None,
                    include_aug = True) -> Pipeline:

    max_one_hot: int = 16
    max_rows = config.AUG_MAX_ROWS
    max_cols = config.AUG_MAX_COLS

    input_val = 'steps.{}.produce'

    # create the basic pipeline
    tabular_pipeline = Pipeline()
    tabular_pipeline.add_input(name='inputs')
    previous_step = 0

    # get the dataset shape
    _, df = base_utils.get_tabular_resource(dataset, None)
    df_rows = df.shape[0]
    df_cols = df.shape[1]

    # check join if requested
    join_result: datamart.DatamartSearchResult = None
    if include_aug:
       join_result, df_cols, df_rows = _query_datamart(keywords, dataset)
    include_aug = join_result != None

    if include_aug:
        # Augment dataset - currently just picks the first query result
        step = PrimitiveStep(primitive_description=DataMartAugmentPrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
        step.add_output('produce')
        step.add_hyperparameter('system_identifier', ArgumentType.VALUE, "NYU")
        step.add_hyperparameter('search_result', ArgumentType.VALUE, join_result.serialize())
        tabular_pipeline.add_step(step)
    else:
        logger.warn("Datamart did not return result for input dataset - proceeding with baseline dataset")

    # Denormalize
    data_ref = ""
    if include_aug:
        data_ref = input_val.format(previous_step)
        previous_step += 1
    else:
        data_ref = 'inputs.0'
    step = PrimitiveStep(primitive_description=DenormalizePrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=data_ref)
    step.add_output('produce')
    tabular_pipeline.add_step(step)

     # if join rows is greater than max rows, sample
    if df_rows > max_rows:
        step = PrimitiveStep(primitive_description=DatasetSamplePrimitive.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('sample_size', ArgumentType.VALUE, max_rows)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # extract dataframe from dataset
    step = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # profile columns since datamart profiling can lead to errors
    # step = PrimitiveStep(primitive_description=SimpleProfilerPrimitive.metadata.query())
    # step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    # step.add_output('produce')
    # tabular_pipeline.add_step(step)
    # previous_step += 1

    step = PrimitiveStep(primitive_description=Simon.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('overwrite', ArgumentType.VALUE, True)
    step.add_hyperparameter('statistical_classification', ArgumentType.VALUE, True)
    step.add_hyperparameter('multi_label_classification', ArgumentType.VALUE, False)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # replace text fields with categorical - work around for text encoder failing on empty strings
    step = PrimitiveStep(primitive_description=ReplaceSemanticTypesPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    from_types = ('http://schema.org/Text',)
    to_types = ('https://metadata.datadrivendiscovery.org/types/CategoricalData',)
    step.add_hyperparameter('from_semantic_types', ArgumentType.VALUE, from_types)
    step.add_hyperparameter('to_semantic_types', ArgumentType.VALUE, to_types)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # replace int fields with float - workaround for column parser failing when a column with semantic type of int contains
    # a floating point value
    # step = PrimitiveStep(primitive_description=ReplaceSemanticTypesPrimitive.metadata.query())
    # step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    # step.add_output('produce')
    # from_types = ('http://schema.org/Integer',)
    # to_types = ('https://schema.org/Float',)
    # step.add_hyperparameter('from_semantic_types', ArgumentType.VALUE, from_types)
    # step.add_hyperparameter('to_semantic_types', ArgumentType.VALUE, to_types)
    # tabular_pipeline.add_step(step)
    # previous_step += 1

    # Parse columns.
    step = PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    semantic_types = ('http://schema.org/Boolean', 'http://schema.org/Integer', 'http://schema.org/Float',
                      'https://metadata.datadrivendiscovery.org/types/FloatVector')
    step.add_hyperparameter('parse_semantic_types', ArgumentType.VALUE, semantic_types)
    tabular_pipeline.add_step(step)
    previous_step += 1
    parse_step = previous_step

    # Extract attributes
    step = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step.add_output('produce')
    step.add_hyperparameter('semantic_types', ArgumentType.VALUE, ('https://metadata.datadrivendiscovery.org/types/Attribute',))
    tabular_pipeline.add_step(step)
    previous_step += 1
    attributes_step = previous_step

    # Extract targets
    step = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step.add_output('produce')
    target_types = ('https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
    step.add_hyperparameter('semantic_types', ArgumentType.VALUE, target_types)
    tabular_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    # Append date enricher.  Looks for date columns and normalizes them.
    step = PrimitiveStep(primitive_description=EnrichDatesPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(attributes_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Append singleton replacer.  Looks for categorical values that only occur once in a column and replace them with a flag.
    step = PrimitiveStep(primitive_description=ReplaceSingletonsPrimitive.metadata.query())
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
    step.add_hyperparameter('min_binary', ArgumentType.VALUE, max_one_hot)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # There shouldn't be any strings by this point, but if something has been mistyped in metadata it could still be
    # hanging around.  We'll pull it out now.
    step = PrimitiveStep(primitive_description=ExtractColumnsByStructuralTypesPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('negate', ArgumentType.VALUE, True)
    step.add_hyperparameter('structural_types', ArgumentType.VALUE, (str,))
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Adds SK learn missing value indicator
    step = PrimitiveStep(primitive_description=SKMissingIndicator.SKMissingIndicator.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('use_semantic_types', ArgumentType.VALUE, False)
    step.add_hyperparameter('return_result', ArgumentType.VALUE, 'append')
    step.add_hyperparameter('error_on_new', ArgumentType.VALUE, False)
    step.add_hyperparameter('error_on_no_input', ArgumentType.VALUE, False)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Adds SK learn simple imputer
    step = PrimitiveStep(primitive_description=SKImputer.SKImputer.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('use_semantic_types', ArgumentType.VALUE, True)
    step.add_hyperparameter('error_on_no_input', ArgumentType.VALUE, False)
    step.add_hyperparameter('return_result', ArgumentType.VALUE, 'replace')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # PCA step
    if df_cols > max_cols:
        step = PrimitiveStep(primitive_description=SKPCA.SKPCA.metadata.query())
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_output('produce')
        step.add_hyperparameter('n_components', ArgumentType.VALUE, max_cols)
        step.add_hyperparameter('error_on_no_input', ArgumentType.VALUE, False)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # Generates a random forest ensemble model.
    step = PrimitiveStep(primitive_description=EnsembleForestPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(target_step))
    step.add_output('produce')
    step.add_hyperparameter('metric', ArgumentType.VALUE, metric)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # # convert predictions to expected format
    step = PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step.add_output('produce')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    tabular_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return tabular_pipeline

def _query_datamart(keywords: List[Any], dataset: Optional[container.Dataset]) -> Tuple[datamart.DatamartSearchResult, int, int]:
     # Search NYU DataMart using dataset and keyword
    URL = os.environ['DATAMART_URL_NYU']
    client = datamart_nyu.RESTDatamart(URL)

    # extract the keywords from the data aug info
    keywords_list: Set[str] = set()
    for keywords_entry in keywords:
        keywords_list.update(keywords_entry['keywords'])

    query = datamart.DatamartQuery(
        keywords = list(keywords_list),
        variables = []
    )
    cursor = client.search_with_data(query, supplied_data=dataset)
    results = cursor.get_next_page()

    print(f'keywords:\n\n {keywords_list}', file=sys.__stdout__)
    _print_results(results)

    # sometimes joins 500 with no explanation - we will try joins until one works
    for result in results:
        error = False
        try:
            join = result.augment(supplied_data=dataset)
        except Exception as e:
            join_with = result.get_json_metadata()['metadata']['name']
            join_col = result.get_json_metadata()['augmentation']['right_columns_names']
            logger.warn(f'Datamart failed to augment using dataset {join_with} column {join_col} - trying next result')
            logger.exception(e)
            error = True
        if not error:
            _, joined_df = base_utils.get_tabular_resource(join, None)
            joined_shape = joined_df.shape
            return result, joined_shape[1], joined_shape[0]
    return (None, 0, 0)

def _print_results(results):
    if not results:
        return
    for result in results:
        print(result.score(), file=sys.__stdout__)
        print(result.get_json_metadata()['metadata']['name'], file=sys.__stdout__)
        if (result.get_augment_hint()):
            print("Left Columns: %s" %
                  str(result.get_json_metadata()['augmentation']['left_columns_names']), file=sys.__stdout__)
            print("Right Columns: %s" %
                  str(result.get_json_metadata()['augmentation']['right_columns_names']), file=sys.__stdout__)
        else:
            print(result.id(), file=sys.__stdout__)
        print("-------------------", file=sys.__stdout__)