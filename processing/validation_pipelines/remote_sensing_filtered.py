from typing import Optional
from distil.primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from common_primitives.simple_profiler import SimpleProfilerPrimitive

from d3m import utils
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.primitives.remote_sensing.remote_sensing_pretrained import (
    RemoteSensingPretrained,
)

from distil.primitives.ranked_linear_svc import RankedLinearSVCPrimitive
from distil.primitives.ensemble_forest import EnsembleForestPrimitive
from distil.primitives.satellite_image_loader import (
    DataFrameSatelliteImageLoaderPrimitive,
)
from distil.primitives.vector_filter import VectorBoundsFilterPrimitive

# Overall implementation relies on passing the entire dataset through the pipeline, with the primitives
# identifying columns to operate on based on type.  Alternative implementation (that better lines up with
# D3M approach, but generates more complex pipelines) would be to extract sub-sets by semantic type using
# a common primitive, apply the type-specific primitive to the sub-set, and then merge the changes
# (replace or join) back into the original data.
def create_pipeline(
    metric: str,
    min_meta: bool = False,
    grid_search: bool = False,
    batch_size: int = 128,
    svc: bool = False,
    n_jobs: int = -1,
    resolver: Optional[Resolver] = None,
) -> Pipeline:
    input_val = "steps.{}.produce"
    # create the basic pipeline
    image_pipeline = Pipeline()
    image_pipeline.add_input(name="inputs")

    # step 0 - denormalize dataframe (N.B.: injects semantic type information)
    step = PrimitiveStep(
        primitive_description=DenormalizePrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step.add_output("produce")
    image_pipeline.add_step(step)
    previous_step = 0

    # step 1 - extract dataframe from dataset
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
    image_pipeline.add_step(step)
    previous_step += 1

    # step 2 - read images
    step = PrimitiveStep(
        primitive_description=DataFrameSatelliteImageLoaderPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    step.add_hyperparameter("return_result", ArgumentType.VALUE, "replace")
    step.add_hyperparameter("n_jobs", ArgumentType.VALUE, 1)
    image_pipeline.add_step(step)
    previous_step += 1
    image_step = previous_step

    if min_meta:
        # Profile columns.
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
        image_pipeline.add_step(step)
        previous_step += 1

    # step 3 - parse columns
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
    step.add_hyperparameter("parsing_semantics", ArgumentType.VALUE, semantic_types)
    image_pipeline.add_step(step)
    previous_step += 1
    parse_step = previous_step

    step = PrimitiveStep(
        primitive_description=VectorBoundsFilterPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    step.add_hyperparameter(
        "row_indices_list",
        ArgumentType.VALUE,
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]],
    )
    step.add_hyperparameter("mins", ArgumentType.VALUE, [-8.5, -9])
    step.add_hyperparameter("maxs", ArgumentType.VALUE, [40, 45])
    image_pipeline.add_step(step)
    previous_step += 1
    # filter_step = previous_step

    # step 4 - extract attributes
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
        ("http://schema.org/ImageObject",),
    )
    image_pipeline.add_step(step)
    previous_step += 1
    attributes_step = previous_step

    # step 5 - extract targets
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
    image_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    # step 6 - featurize imagery
    step = PrimitiveStep(
        primitive_description=RemoteSensingPretrained.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(attributes_step),
    )
    step.add_output("produce")
    step.add_hyperparameter("batch_size", ArgumentType.VALUE, batch_size)
    image_pipeline.add_step(step)
    previous_step += 1

    # step 7 - Generates a linear sv or random forest model.
    if svc:
        # use linear svc
        step = PrimitiveStep(
            primitive_description=RankedLinearSVCPrimitive.metadata.query(),
            resolver=resolver,
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(previous_step),
        )
        step.add_argument(
            name="outputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(target_step),
        )
        step.add_output("produce")
        step.add_hyperparameter("scaling", ArgumentType.VALUE, "unit_norm")
        step.add_hyperparameter("rank_confidences", ArgumentType.VALUE, False)
    else:
        # use random forest
        step = PrimitiveStep(
            primitive_description=EnsembleForestPrimitive.metadata.query(),
            resolver=resolver,
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(previous_step),
        )
        step.add_argument(
            name="outputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(target_step),
        )
        step.add_output("produce")
        step.add_hyperparameter("n_jobs", ArgumentType.VALUE, n_jobs)
        step.add_hyperparameter("metric", ArgumentType.VALUE, metric)
        step.add_hyperparameter("grid_search", ArgumentType.VALUE, grid_search)
        step.add_hyperparameter("compute_confidences", ArgumentType.VALUE, False)

    image_pipeline.add_step(step)
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
    step.add_hyperparameter("use_columns", ArgumentType.VALUE, [0, 1])
    image_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    image_pipeline.add_output(
        name="output", data_reference=input_val.format(previous_step)
    )

    return image_pipeline, []
