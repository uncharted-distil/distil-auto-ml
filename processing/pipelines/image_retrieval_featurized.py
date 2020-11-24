from typing import Optional
from distil.primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from common_primitives.extract_columns_structural_types import (
    ExtractColumnsByStructuralTypesPrimitive,
)
from common_primitives.add_semantic_types import AddSemanticTypesPrimitive

from d3m import utils
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from distil.primitives.satellite_image_loader import (
    DataFrameSatelliteImageLoaderPrimitive,
)
from distil.primitives.list_to_dataframe import ListEncoderPrimitive
from d3m.primitives.similarity_modeling.iterative_labeling import ImageRetrieval
from d3m.primitives.remote_sensing.remote_sensing_pretrained import (
    RemoteSensingPretrained,
)
from dsbox.datapreprocessing.cleaner.iterative_regression import (
    IterativeRegressionImputation,
)


def create_pipeline(
    metric: str,
    min_meta: bool = False,
    batch_size: int = 256,
    gem_p: int = 1,
    reduce_dimension: int = 32,
    n_jobs: int = -1,
    resolver: Optional[Resolver] = None,
) -> Pipeline:
    input_val = "steps.{}.produce"
    # create the basic pipeline
    image_pipeline = Pipeline()
    image_pipeline.add_input(name="inputs")
    image_pipeline.add_input(name="annotations")

    # step 1 - extract dataframe from dataset
    step = PrimitiveStep(
        primitive_description=DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="inputs.0",
    )
    step.add_output("produce")
    image_pipeline.add_step(step)
    previous_step = 0

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
            "https://metadata.datadrivendiscovery.org/types/Attribute",
            "https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey",
        ),
    )
    image_pipeline.add_step(step)
    previous_step += 1
    attributes_step = previous_step

    # step 6
    step = PrimitiveStep(
        primitive_description=DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="inputs.1",
    )
    step.add_output("produce")
    image_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    # List encoder to get from vectors to columns
    # step = PrimitiveStep(
    #     primitive_description=ListEncoderPrimitive.metadata.query(),
    #     resolver=resolver,
    # )
    # step.add_argument(
    #     name="inputs",
    #     argument_type=ArgumentType.CONTAINER,
    #     data_reference=input_val.format(attributes_step),
    # )
    # step.add_output("produce")
    # image_pipeline.add_step(step)
    # previous_step += 1

    # Extract floats to ensure that we're only passing valid data into the learner
    step = PrimitiveStep(
        primitive_description=ExtractColumnsByStructuralTypesPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(attributes_step),
    )
    step.add_output("produce")
    step.add_hyperparameter(
        "structural_types",
        ArgumentType.VALUE,
        ("int", "float", "numpy.float32", "numpy.float64"),
    )
    image_pipeline.add_step(step)
    previous_step += 1

    # Impute any missing data
    # step = PrimitiveStep(
    #     primitive_description=IterativeRegressionImputation.metadata.query(),
    #     resolver=resolver,
    # )
    # step.add_argument(
    #     name="inputs",
    #     argument_type=ArgumentType.CONTAINER,
    #     data_reference=input_val.format(previous_step),
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
        data_reference=input_val.format(previous_step),
    )
    step.add_argument(
        name="outputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(target_step),
    )
    step.add_output("produce")
    step.add_hyperparameter(name="gem_p", argument_type=ArgumentType.VALUE, data=gem_p)
    step.add_hyperparameter(
        name="reduce_dimension", argument_type=ArgumentType.VALUE, data=reduce_dimension
    )
    image_pipeline.add_step(step)
    previous_step += 1

    step = PrimitiveStep(
        primitive_description=AddSemanticTypesPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    step.add_hyperparameter("columns", ArgumentType.VALUE, [1])
    step.add_hyperparameter(
        "semantic_types",
        ArgumentType.VALUE,
        [
            "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
            "https://metadata.datadrivendiscovery.org/types/Score",
        ],
    )
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
