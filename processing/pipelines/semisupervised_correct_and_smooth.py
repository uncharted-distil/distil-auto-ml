from typing import Optional

from distil.primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.primitives.data_cleaning.column_type_profiler import Simon
from d3m.primitives.data_cleaning.imputer import SKlearn as SKImputer
from d3m.primitives.semisupervised_classification.iterative_labeling import (
    CorrectAndSmooth,
)
from d3m.primitives.semisupervised_classification.iterative_labeling import (
    CorrectAndSmooth,
)


def create_pipeline(
    metric: str,
    normalize_features: bool = True,
    resolver: Optional[Resolver] = None,
) -> Pipeline:
    input_val = "steps.{}.produce"
    tune_steps = []

    # create the basic pipeline
    semisupervised_pipeline = Pipeline()
    semisupervised_pipeline.add_input(name="inputs")

    step = PrimitiveStep(
        primitive_description=DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step.add_output("produce")
    semisupervised_pipeline.add_step(step)
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
    semisupervised_pipeline.add_step(step)
    previous_step += 1
    profiler_step = previous_step

    step = PrimitiveStep(
        primitive_description=ColumnParserPrimitive.metadata.query(),
        resolver=resolver,
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
    step.add_hyperparameter("parsing_semantics", ArgumentType.VALUE, semantic_types)
    semisupervised_pipeline.add_step(step)
    previous_step += 1

    step = PrimitiveStep(primitive_description=SKImputer.metadata.query())
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    step.add_hyperparameter("return_result", ArgumentType.VALUE, "replace")
    step.add_hyperparameter("use_semantic_types", ArgumentType.VALUE, True)
    semisupervised_pipeline.add_step(step)
    previous_step += 1

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
    semisupervised_pipeline.add_step(step)
    previous_step += 1
    attributes_step = previous_step

    step = PrimitiveStep(
        primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(attributes_step),
    )
    step.add_output("produce")
    step.add_hyperparameter(
        "semantic_types",
        ArgumentType.VALUE,
        (
            "http://schema.org/Integer",
            "http://schema.org/Float",
        ),
    )
    semisupervised_pipeline.add_step(step)
    previous_step += 1
    inputs_step = previous_step

    step = PrimitiveStep(
        primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(profiler_step),
    )
    step.add_output("produce")
    step.add_hyperparameter(
        "semantic_types",
        ArgumentType.VALUE,
        ("https://metadata.datadrivendiscovery.org/types/Target",),
    )
    semisupervised_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    step = PrimitiveStep(primitive_description=CorrectAndSmooth.metadata.query())
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(inputs_step),
    )
    step.add_argument(
        name="outputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(target_step),
    )
    step.add_hyperparameter(
        "normalize_features",
        ArgumentType.VALUE,
        normalize_features,
    )
    step.add_output("produce")
    semisupervised_pipeline.add_step(step)
    previous_step += 1

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
        data_reference=input_val.format(profiler_step),
    )
    step.add_output("produce")
    semisupervised_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    semisupervised_pipeline.add_output(
        name="output", data_reference=input_val.format(previous_step)
    )

    return (semisupervised_pipeline, [])
