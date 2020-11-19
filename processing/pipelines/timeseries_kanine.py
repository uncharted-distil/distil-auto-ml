from typing import Optional

from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.primitives.time_series_classification.k_neighbors import Kanine
from distil.primitives.time_series_formatter import TimeSeriesFormatterPrimitive


def create_pipeline(
    metric: str, datasplit: bool = False, resolver: Optional[Resolver] = None
) -> Pipeline:
    input_val = "steps.{}.produce"
    tune_steps = []

    # create the basic pipeline
    pipeline = Pipeline()
    pipeline.add_input(name="inputs")

    # step 0 - flatten the timeseries if necessary
    step = PrimitiveStep(
        primitive_description=TimeSeriesFormatterPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step.add_output("produce")
    pipeline.add_step(step)
    previous_step = 0

    # extract dataframe from dataset
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
    pipeline.add_step(step)
    previous_step += 1

    # run profiler
    step = PrimitiveStep(
        primitive_description=SimpleProfilerPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    pipeline.add_step(step)
    previous_step += 1
    tune_steps.append(previous_step)

    # Parse columns.
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
    pipeline.add_step(step)
    previous_step += 1

    # Extract attributes
    step = PrimitiveStep(
        primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    step.add_hyperparameter(
        "semantic_types",
        ArgumentType.VALUE,
        ("https://metadata.datadrivendiscovery.org/types/Attribute",),
    )
    pipeline.add_step(step)
    previous_step += 1
    attributes_step = previous_step

    # extract dataframe from dataset
    step = PrimitiveStep(
        primitive_description=DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step.add_output("produce")
    pipeline.add_step(step)
    previous_step += 1

    # run profiler
    step = PrimitiveStep(
        primitive_description=SimpleProfilerPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    pipeline.add_step(step)
    previous_step += 1
    profile_step = previous_step

    # Extract targets
    step = PrimitiveStep(
        primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    target_types = (
        "https://metadata.datadrivendiscovery.org/types/Target",
        "https://metadata.datadrivendiscovery.org/types/TrueTarget",
    )
    step.add_hyperparameter("semantic_types", ArgumentType.VALUE, target_types)
    pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    # kanine classification
    step = PrimitiveStep(
        primitive_description=Kanine.metadata.query(), resolver=resolver
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
    pipeline.add_step(step)
    previous_step += 1
    tune_steps.append(previous_step)

    # Step 7: construct predictions
    step_6 = PrimitiveStep(
        ConstructPredictionsPrimitive.metadata.query(), resolver=resolver
    )
    step_6.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step_6.add_argument(
        name="reference",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(profile_step),
    )
    step_6.add_output("produce")
    pipeline.add_step(step_6)
    previous_step += 1

    # Adding output step to the pipeline
    pipeline.add_output(name="output", data_reference=input_val.format(previous_step))

    return (pipeline, tune_steps)
