from typing import Optional
from distil.primitives.ensemble_forest import EnsembleForestPrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from common_primitives.grouping_field_compose import GroupingFieldComposePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from common_primitives.xgboost_gbtree import XGBoostGBTreeClassifierPrimitive
from common_primitives.xgboost_regressor import XGBoostGBTreeRegressorPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from processing.metrics import (
    regression_metrics,
    classification_metrics,
    confidence_metrics,
)
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.pipeline import Resolver


def create_pipeline(
    metric: str,
    use_boost: bool = False,
    grid_search=False,
    n_jobs: int = -1,
    compute_confidences=False,
    resolver: Optional[Resolver] = None,
) -> Pipeline:
    previous_step = 0
    tune_steps = []
    input_val = "steps.{}.produce"

    ts_tabular_pipeline = Pipeline()
    ts_tabular_pipeline.add_input(name="inputs")

    # step 0 - Extract dataframe from dataset
    step = PrimitiveStep(
        primitive_description=DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step.add_output("produce")
    ts_tabular_pipeline.add_step(step)

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
    ts_tabular_pipeline.add_step(step)
    previous_step += 1

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
        "http://schema.org/Boolean",
        "http://schema.org/Integer",
        "http://schema.org/Float",
        "https://metadata.datadrivendiscovery.org/types/FloatVector",
        "http://schema.org/DateTime",
    )
    step.add_hyperparameter("parse_semantic_types", ArgumentType.VALUE, semantic_types)
    ts_tabular_pipeline.add_step(step)
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
    ts_tabular_pipeline.add_step(step)
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
    ts_tabular_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    if use_boost:
        if metric in regression_metrics:
            step = PrimitiveStep(
                primitive_description=XGBoostGBTreeRegressorPrimitive.metadata.query(),
                resolver=resolver,
            )
        elif metric in classification_metrics and metric not in confidence_metrics:
            # xgboost classifier doesn't support probability generation so no support for confidence-based metrics
            step = PrimitiveStep(
                primitive_description=XGBoostGBTreeClassifierPrimitive.metadata.query(),
                resolver=resolver,
            )
    else:
        step = PrimitiveStep(
            primitive_description=EnsembleForestPrimitive.metadata.query(),
            resolver=resolver,
        )
        step.add_hyperparameter("grid_search", ArgumentType.VALUE, grid_search)
        step.add_hyperparameter("small_dataset_fits", ArgumentType.VALUE, 1)
        step.add_hyperparameter(
            "compute_confidences", ArgumentType.VALUE, compute_confidences
        )

    step.add_hyperparameter("n_jobs", ArgumentType.VALUE, n_jobs)
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

    if not use_boost:
        step.add_hyperparameter("metric", ArgumentType.VALUE, metric)

    ts_tabular_pipeline.add_step(step)
    previous_step += 1
    tune_steps.append(previous_step)

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
    ts_tabular_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    ts_tabular_pipeline.add_output(
        name="output", data_reference=input_val.format(previous_step)
    )

    return (ts_tabular_pipeline, tune_steps)
