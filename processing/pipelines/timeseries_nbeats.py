from typing import Optional
from distil.primitives.time_series_formatter import TimeSeriesFormatterPrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from common_primitives.grouping_field_compose import GroupingFieldComposePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.primitives.time_series_forecasting.feed_forward_neural_net import NBEATS
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.pipeline import Resolver


def create_pipeline(
    metric: str, grouping_compose: bool = True, resolver: Optional[Resolver] = None
) -> Pipeline:
    previous_step = 0
    tune_steps = []
    input_val = "steps.{}.produce"

    nbeats_pipeline = Pipeline()
    nbeats_pipeline.add_input(name="inputs")

    # step 0 - flatten the timeseries if necessary
    step = PrimitiveStep(
        primitive_description=TimeSeriesFormatterPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step.add_output("produce")
    nbeats_pipeline.add_step(step)

    # step 0 - Extract dataframe from dataset
    step = PrimitiveStep(
        primitive_description=DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    nbeats_pipeline.add_step(step)
    previous_step += 1

    # step 1
    step = PrimitiveStep(
        primitive_description=SimpleProfilerPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    # step.add_hyperparameter(
    #     "categorical_max_ratio_distinct_values", ArgumentType.VALUE, 0
    # )
    step.add_output("produce")
    nbeats_pipeline.add_step(step)
    previous_step += 1

    # step 2
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
        "http://schema.org/Boolean",
        "http://schema.org/Integer",
        "http://schema.org/Float",
        "https://metadata.datadrivendiscovery.org/types/FloatVector",
        "http://schema.org/DateTime",
    )
    step.add_hyperparameter("parse_semantic_types", ArgumentType.VALUE, semantic_types)
    nbeats_pipeline.add_step(step)
    previous_step += 1
    parse_step = previous_step

    if grouping_compose:
        step = PrimitiveStep(
            primitive_description=GroupingFieldComposePrimitive.metadata.query(),
            resolver=resolver,
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(previous_step),
        )
        step.add_output("produce")
        nbeats_pipeline.add_step(step)
        previous_step += 1
        parse_step = previous_step

    # step 3
    step = PrimitiveStep(
        primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(parse_step),
    )
    step.add_hyperparameter(
        name="semantic_types",
        argument_type=ArgumentType.VALUE,
        data=[
            "https://metadata.datadrivendiscovery.org/types/Attribute",
            "https://metadata.datadrivendiscovery.org/types/GroupingKey",
        ],
    )
    step.add_output("produce")
    nbeats_pipeline.add_step(step)
    previous_step += 1
    attribute_step = previous_step

    # step 4
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
    nbeats_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    # step 5
    step = PrimitiveStep(
        primitive_description=NBEATS.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(attribute_step),
    )
    step.add_argument(
        name="outputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(target_step),
    )
    step.add_output("produce")
    # step.add_hyperparameter("epochs", ArgumentType.VALUE, 1)
    # step.add_hyperparameter("steps_per_epoch", ArgumentType.VALUE, 1)
    # step.add_hyperparameter("num_estimators", ArgumentType.VALUE, 1)
    # step.add_hyperparameter("prediction_length", ArgumentType.VALUE, 10)
    # step.add_hyperparameter("num_context_lengths", ArgumentType.VALUE, 2)
    step.add_hyperparameter("nan_padding", ArgumentType.VALUE, False)
    nbeats_pipeline.add_step(step)
    previous_step += 1
    tune_steps.append(previous_step)

    # step 6
    step = PrimitiveStep(
        primitive_description=ConstructPredictionsPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_argument(
        name="reference",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(parse_step),
    )
    step.add_output("produce")
    nbeats_pipeline.add_step(step)
    previous_step += 1

    nbeats_pipeline.add_output(
        name="output", data_reference=input_val.format(previous_step)
    )

    return (nbeats_pipeline, tune_steps)
