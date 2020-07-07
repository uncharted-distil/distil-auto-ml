from typing import Optional

from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.primitives.time_series_forecasting.vector_autoregression import VAR

def create_pipeline(metric: str, resolver: Optional[Resolver] = None) -> Pipeline:
    previous_step = 0
    tune_steps = []
    input_val = "steps.{}.produce"

    # create the basic pipeline
    var_pipeline = Pipeline()
    var_pipeline.add_input(name="inputs")

    # step 0 - Extract dataframe from dataset
    step = PrimitiveStep(
        primitive_description=DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step.add_output("produce")
    var_pipeline.add_step(step)

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
    var_pipeline.add_step(step)
    previous_step += 1
    # tune_steps.append(previous_step)

    # step 1 - Parse columns.
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
    var_pipeline.add_step(step)
    previous_step += 1
    parse_step = previous_step

    # Extract attributes
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
        (
            "https://metadata.datadrivendiscovery.org/types/Attribute",
            "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
        ),
    )
    var_pipeline.add_step(step)
    previous_step += 1
    attributes_step = previous_step

    # Extract targets
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
    var_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    # step 2 - Vector Auto Regression
    step = PrimitiveStep(primitive_description=VAR.metadata.query(), resolver=resolver)
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
    var_pipeline.add_step(step)
    previous_step += 1

    # step 3 - Generate predictions output
    step = PrimitiveStep(
        primitive_description=ConstructPredictionsPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                      data_reference=input_val.format(previous_step))
    step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER,
                      data_reference=input_val.format(parse_step))
    step.add_output('produce')
    var_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    var_pipeline.add_output(
        name="output", data_reference=input_val.format(previous_step)
    )
    tune_steps.append(previous_step)

    return (var_pipeline, tune_steps)
