from typing import Optional
from datetime import datetime

from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from common_primitives.datetime_field_compose import DatetimeFieldComposePrimitive
from common_primitives.datetime_range_filter import DatetimeRangeFilterPrimitive
from common_primitives.term_filter import TermFilterPrimitive
from common_primitives.regex_filter import RegexFilterPrimitive
from common_primitives.numeric_range_filter import NumericRangeFilterPrimitive
from common_primitives.xgboost_gbtree import XGBoostGBTreeClassifierPrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive

from d3m import utils
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver


def create_pipeline(metric: str,
                    resolver: Optional[Resolver] = None) -> Pipeline:
    input_val = 'steps.{}.produce'
    common_pipeline = Pipeline()
    common_pipeline.add_input(name='inputs')
    tune_steps = []

    step = PrimitiveStep(
        primitive_description=DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference='inputs.0'
    )
    step.add_output("produce")
    common_pipeline.add_step(step)
    previous_step = 0

    step = PrimitiveStep(primitive_description=SimpleProfilerPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                        data_reference=input_val.format(previous_step))
    step.add_output('produce')
    common_pipeline.add_step(step)
    previous_step += 1

    step = PrimitiveStep(primitive_description=DatetimeFieldComposePrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('columns', ArgumentType.VALUE, [4, 5, 22])
    step.add_hyperparameter('join_char', ArgumentType.VALUE, '-')
    step.add_hyperparameter('output_name', ArgumentType.VALUE, 'timestamp')
    common_pipeline.add_step(step)
    previous_step += 1

    step = PrimitiveStep(primitive_description=DatetimeRangeFilterPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('column', ArgumentType.VALUE, 3)
    step.add_hyperparameter('min', ArgumentType.VALUE, datetime(2018, 1, 1))
    step.add_hyperparameter('max', ArgumentType.VALUE, datetime(2018, 12, 5))
    step.add_hyperparameter('strict', ArgumentType.VALUE, True)
    step.add_hyperparameter('inclusive', ArgumentType.VALUE, True)
    common_pipeline.add_step(step)
    previous_step += 1

    step = PrimitiveStep(primitive_description=TermFilterPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('column', ArgumentType.VALUE, 14)
    step.add_hyperparameter('inclusive', ArgumentType.VALUE, True)
    step.add_hyperparameter('terms', ArgumentType.VALUE, ["Middle East"])
    step.add_hyperparameter('match_whole', ArgumentType.VALUE, True)
    common_pipeline.add_step(step)
    previous_step += 1

    step = PrimitiveStep(primitive_description=RegexFilterPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('column', ArgumentType.VALUE, 25)
    step.add_hyperparameter('inclusive', ArgumentType.VALUE, True)
    step.add_hyperparameter('regex', ArgumentType.VALUE, 'Syria')
    common_pipeline.add_step(step)
    previous_step += 1

    step = PrimitiveStep(primitive_description=NumericRangeFilterPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    step.add_hyperparameter('column', ArgumentType.VALUE, 22)
    step.add_hyperparameter('min', ArgumentType.VALUE, 0)
    step.add_hyperparameter('max', ArgumentType.VALUE, 100)
    step.add_hyperparameter('strict', ArgumentType.VALUE, True)
    step.add_hyperparameter('inclusive', ArgumentType.VALUE, True)
    common_pipeline.add_step(step)
    previous_step += 1

    step = PrimitiveStep(
        primitive_description=ColumnParserPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    semantic_types = (
        "http://schema.org/Integer",
        "http://schema.org/Float",
        "https://metadata.datadrivendiscovery.org/types/FloatVector",
    )
    step.add_hyperparameter("parse_semantic_types", ArgumentType.VALUE, semantic_types)
    common_pipeline.add_step(step)
    previous_step += 1
    parse_step = previous_step

    step = PrimitiveStep(
        primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(parse_step),
    )
    step.add_output("produce")
    step.add_hyperparameter(
        "semantic_types",
        ArgumentType.VALUE,
        ("https://metadata.datadrivendiscovery.org/types/Attribute",),
    )
    common_pipeline.add_step(step)
    previous_step += 1
    attributes_step = previous_step

    step = PrimitiveStep(
        primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(parse_step),
    )
    step.add_output("produce")
    target_types = (
        "https://metadata.datadrivendiscovery.org/types/Target",
        "https://metadata.datadrivendiscovery.org/types/TrueTarget",
    )
    step.add_hyperparameter("semantic_types", ArgumentType.VALUE, target_types)
    common_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    step = PrimitiveStep(
        primitive_description=XGBoostGBTreeClassifierPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(attributes_step),
    )
    step.add_argument(
        name="outputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(target_step),
    )
    step.add_output("produce")
    common_pipeline.add_step(step)
    previous_step += 1
    tune_steps.append(previous_step)

    common_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return common_pipeline, tune_steps
