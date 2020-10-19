from typing import Optional

from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.add_semantic_types import AddSemanticTypesPrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from common_primitives.xgboost_gbtree import XGBoostGBTreeClassifierPrimitive
from common_primitives.simple_profiler import SimpleProfilerPrimitive

from distil.primitives.mi_ranking import MIRankingPrimitive
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver


def create_pipeline(metric: str, resolver: Optional[Resolver] = None) -> Pipeline:
    input_val = "steps.{}.produce"

    mi_ranking_pipeline = Pipeline()
    mi_ranking_pipeline.add_input(name="inputs")

    # convert to a dataframe
    step = PrimitiveStep(
        primitive_description=DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step.add_output("produce")
    mi_ranking_pipeline.add_step(step)
    previous_step = 0

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
    mi_ranking_pipeline.add_step(step)
    previous_step += 1

    # parse columns into associated data frame types
    step = PrimitiveStep(
        primitive_description=ColumnParserPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_hyperparameter("parse_categorical_target_columns", ArgumentType.VALUE, True)
    step.add_output("produce")
    mi_ranking_pipeline.add_step(step)
    previous_step += 1
    parse_step = previous_step

    # Perform MI ranking and write reuslts to metadata
    step = PrimitiveStep(
        primitive_description=MIRankingPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    step.add_hyperparameter("target_col_index", ArgumentType.VALUE, 18)
    step.add_hyperparameter("return_as_metadata", ArgumentType.VALUE, True)
    mi_ranking_pipeline.add_step(step)
    previous_step += 1
    ranking_step = previous_step

    # Extract attributes
    step = PrimitiveStep(
        primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(ranking_step),
    )
    step.add_output("produce")
    step.add_hyperparameter(
        "semantic_types",
        ArgumentType.VALUE,
        ("https://metadata.datadrivendiscovery.org/types/Attribute",),
    )
    mi_ranking_pipeline.add_step(step)
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
        data_reference=input_val.format(ranking_step),
    )
    step.add_output("produce")
    target_types = (
        "https://metadata.datadrivendiscovery.org/types/Target",
        "https://metadata.datadrivendiscovery.org/types/TrueTarget",
    )
    step.add_hyperparameter("semantic_types", ArgumentType.VALUE, target_types)
    mi_ranking_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    # use random forest
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

    mi_ranking_pipeline.add_step(step)
    previous_step += 1

    # step 8 - convert predictions to expected format
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
    mi_ranking_pipeline.add_step(step)
    previous_step += 1

    mi_ranking_pipeline.add_output(
        name="output", data_reference=input_val.format(previous_step)
    )

    return (mi_ranking_pipeline, [])
