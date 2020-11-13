from typing import Optional
from distil.primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)

from d3m import utils
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from distil.primitives.satellite_image_loader import (
    DataFrameSatelliteImageLoaderPrimitive,
)
from d3m.primitives.similarity_modeling.iterative_labeling import ImageRetrieval
from d3m.primitives.remote_sensing.remote_sensing_pretrained import (
    RemoteSensingPretrained,
)


def create_pipeline(
    metric: str,
    min_meta: bool = False,
    batch_size: int = 256,
    gem_p: int = 1,
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
    df_step = previous_step

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
    step.add_hyperparameter("n_jobs", ArgumentType.VALUE, n_jobs)
    image_pipeline.add_step(step)
    previous_step += 1
    image_step = previous_step

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
        (
            "http://schema.org/ImageObject",
            "https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey",
        ),
    )
    image_pipeline.add_step(step)
    previous_step += 1
    attributes_step = previous_step

    # step 5 - featurize imagery
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
    remote_step = previous_step

    # step 6
    # step = PrimitiveStep(
    #     primitive_description=DatasetToDataFramePrimitive.metadata.query(),
    #     resolver=resolver,
    # )
    # step.add_argument(
    #     name="inputs",
    #     argument_type=ArgumentType.CONTAINER,
    #     data_reference="inputs.1",  # input_val.format(image_step),
    # )
    # step.add_output("produce")
    # image_pipeline.add_step(step)
    # previous_step += 1

    # step 7
    step = PrimitiveStep(
        primitive_description=ImageRetrieval.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(remote_step),
    )
    step.add_argument(
        name="outputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(df_step),
    )
    step.add_output("produce")
    step.add_hyperparameter(name="gem_p", argument_type=ArgumentType.VALUE, data=gem_p)
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

    image_pipeline.add_output(
        name="output", data_reference=input_val.format(previous_step)
    )

    return image_pipeline, []
