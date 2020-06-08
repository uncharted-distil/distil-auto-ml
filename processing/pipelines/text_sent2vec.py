from typing import Optional
from d3m.metadata.pipeline import Resolver
from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.primitives.feature_extraction.nk_sent2vec import Sent2Vec
from d3m.primitives.data_cleaning.column_type_profiler import Simon
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.text_reader import TextReaderPrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
from distil.primitives.ensemble_forest import EnsembleForestPrimitive

import logging
logging.basicConfig(level=logging.DEBUG)

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
                    min_meta: bool = False,
                    resolver: Optional[Resolver] = None) -> Pipeline:

    # create the basic pipeline
    input_val = 'steps.{}.produce'
    text_pipeline = Pipeline()
    text_pipeline.add_input(name='inputs')

    # step 0 - denormalize dataframe (injects semantic type information)
    step = PrimitiveStep(primitive_description=DenormalizePrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    text_pipeline.add_step(step)
    previous_step = 0

    # step 1 - extract dataframe from dataset
    step = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    text_pipeline.add_step(step)
    previous_step += 1


    # step 2 - read text
    step = PrimitiveStep(primitive_description=TextReaderPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('use_columns', ArgumentType.VALUE,[0,1])
    step.add_hyperparameter('return_result', ArgumentType.VALUE, 'replace')
    text_pipeline.add_step(step)
    previous_step += 1

    if min_meta:
        step = PrimitiveStep(primitive_description=Simon.metadata.query(), resolver=resolver)
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
        step.add_hyperparameter(name='overwrite', argument_type=ArgumentType.VALUE, data=True)
        step.add_output('produce')
        text_pipeline.add_step(step)
        previous_step += 1


    # step 3 - Parse columns.
    step = PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    semantic_types = ('http://schema.org/Boolean', 'http://schema.org/Integer', 'http://schema.org/Float',
                      'https://metadata.datadrivendiscovery.org/types/FloatVector')
    step.add_hyperparameter('parse_semantic_types', ArgumentType.VALUE, semantic_types)
    text_pipeline.add_step(step)
    previous_step += 1
    parse_step = previous_step


    # step 4 - Extract attributes
    step = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step.add_output('produce')
    step.add_hyperparameter('semantic_types', ArgumentType.VALUE, ('https://metadata.datadrivendiscovery.org/types/Attribute',))
    text_pipeline.add_step(step)
    previous_step += 1
    attibute_step = previous_step

    # step 5 - Extract targets
    step = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step.add_output('produce')
    target_types = ('https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
    step.add_hyperparameter('semantic_types', ArgumentType.VALUE, target_types)
    text_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    # step 6 - Generate feature vectors
    step = PrimitiveStep(primitive_description=Sent2Vec.metadata.query(), resolver=resolver)
    step.add_argument(name="inputs", argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(attibute_step))
    step.add_output("produce")
    text_pipeline.add_step(step)
    previous_step += 1

    # step 6 - generates an RF model.
    step = PrimitiveStep(primitive_description=EnsembleForestPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(target_step))
    step.add_output('produce')
    step.add_hyperparameter('metric', ArgumentType.VALUE, metric)
    step.add_hyperparameter('grid_search', ArgumentType.VALUE, True)
    text_pipeline.add_step(step)
    previous_step += 1

    # step 7 - convert predictions to expected format
    step = PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step.add_output('produce')
    step.add_hyperparameter('use_columns', ArgumentType.VALUE, [0, 1])
    text_pipeline.add_step(step)
    previous_step += 1


    # Adding output step to the pipeline
    text_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return text_pipeline