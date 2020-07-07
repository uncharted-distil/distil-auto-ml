from typing import Optional

from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.dataframe_image_reader import DataFrameImageReaderPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from d3m import utils
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.primitives.data_preprocessing.dataset_sample import (
    Common as DatasetSamplePrimitive,
)
from distil.primitives.ensemble_forest import EnsembleForestPrimitive
from distil.primitives.image_transfer import ImageTransferPrimitive

PipelineContext = utils.Enum(value="PipelineContext", names=["TESTING"], start=1)

def create_pipeline(metric: str,
                    min_meta: bool = False,
                    sample: bool = False,
                    resolver: Optional[Resolver] = None) -> Pipeline:
    input_val = 'steps.{}.produce'
    # create the basic pipeline
    image_pipeline = Pipeline(context=PipelineContext.TESTING)
    image_pipeline.add_input(name="inputs")
    tune_steps = []

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

    # step 1 - sample dataset down, since some take to long to run.
    if sample:
        step = PrimitiveStep(
            primitive_description=DatasetSamplePrimitive.metadata.query(),
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
        primitive_description=DataFrameImageReaderPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    step.add_hyperparameter("use_columns", ArgumentType.VALUE, [0, 1])
    step.add_hyperparameter("return_result", ArgumentType.VALUE, "replace")
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
        "http://schema.org/Boolean",
        "http://schema.org/Integer",
        "http://schema.org/Float",
        "https://metadata.datadrivendiscovery.org/types/FloatVector",
    )
    step.add_hyperparameter("parse_semantic_types", ArgumentType.VALUE, semantic_types)
    image_pipeline.add_step(step)
    previous_step += 1
    parse_step = previous_step

    # step 4 - featurize images
    step = PrimitiveStep(
        primitive_description=ImageTransferPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(parse_step),
    )
    step.add_output("produce")
    image_pipeline.add_step(step)
    previous_step += 1
    input_step = previous_step

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

    # step 6 - Generates a random forest ensemble model.
    step = PrimitiveStep(
        primitive_description=EnsembleForestPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(input_step),
    )
    step.add_argument(
        name="outputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(target_step),
    )
    step.add_output("produce")
    step.add_hyperparameter("metric", ArgumentType.VALUE, metric)
    image_pipeline.add_step(step)
    previous_step += 1
    tune_steps.append(previous_step)

    # step 7 - convert predictions to expected format
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
        data_reference=input_val.format(image_step),
    )
    step.add_output("produce")
    step.add_hyperparameter("use_columns", ArgumentType.VALUE, [0, 1])
    image_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    image_pipeline.add_output(
        name="output", data_reference=input_val.format(previous_step)
    )

    return (image_pipeline, tune_steps)
