from typing import Optional

from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from distil.primitives.audio_reader import AudioDatasetLoaderPrimitive
from distil.primitives.audio_transfer import AudioTransferPrimitive
from distil.primitives.ensemble_forest import EnsembleForestPrimitive


# Overall implementation relies on passing the entire dataset through the pipeline, with the primitives
# identifying columns to operate on based on type.  Alternative implementation (that better lines up with
# D3M approach, but generates more complex pipelines) would be to extract sub-sets by semantic type using
# a common primitive, apply the type-specific primitive to the sub-set, and then merge the changes
# (replace or join) back into the original data.
def create_pipeline(
    metric: str,
    n_jobs: int = -1,
    resolver: Optional[Resolver] = None,
) -> Pipeline:
    previous_step = 0
    input_val = "steps.{}.produce"

    # create the basic pipeline
    audio_pipeline = Pipeline()
    audio_pipeline.add_input(name="inputs")
    tune_steps = []

    # step 0
    step = PrimitiveStep(
        primitive_description=AudioDatasetLoaderPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step.add_hyperparameter("n_jobs", ArgumentType.VALUE, n_jobs)
    step.add_output("produce")
    step.add_output("produce_collection")
    # step.add_hyperparameter('sample', ArgumentType.VALUE, 0.1)
    audio_pipeline.add_step(step)

    # step 1 - parse columns.
    step = PrimitiveStep(
        primitive_description=ColumnParserPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="steps.0.produce",
    )
    step.add_output("produce")
    semantic_types = (
        "http://schema.org/Boolean",
        "http://schema.org/Integer",
        "http://schema.org/Float",
        "https://metadata.datadrivendiscovery.org/types/FloatVector",
    )
    step.add_hyperparameter("parse_semantic_types", ArgumentType.VALUE, semantic_types)
    audio_pipeline.add_step(step)

    # step 2 - Extract targets
    step = PrimitiveStep(
        primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="steps.1.produce",
    )
    step.add_output("produce")
    target_types = (
        "https://metadata.datadrivendiscovery.org/types/Target",
        "https://metadata.datadrivendiscovery.org/types/TrueTarget",
    )
    step.add_hyperparameter("semantic_types", ArgumentType.VALUE, target_types)
    audio_pipeline.add_step(step)

    # step 3 - featurize
    step = PrimitiveStep(
        primitive_description=AudioTransferPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="steps.0.produce_collection",
    )
    step.add_output("produce")
    audio_pipeline.add_step(step)

    # step 4 -- Generates a random forest ensemble model.
    step = PrimitiveStep(
        primitive_description=EnsembleForestPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="steps.3.produce",
    )
    step.add_argument(
        name="outputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="steps.2.produce",
    )
    step.add_output("produce")
    step.add_hyperparameter("metric", ArgumentType.VALUE, metric)
    step.add_hyperparameter("n_jobs", ArgumentType.VALUE, n_jobs)
    audio_pipeline.add_step(step)
    tune_steps.append(4)

    # step 5 - convert predictions to expected format
    step = PrimitiveStep(
        primitive_description=ConstructPredictionsPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="steps.4.produce",
    )
    step.add_argument(
        name="reference",
        argument_type=ArgumentType.CONTAINER,
        data_reference="steps.1.produce",
    )
    step.add_output("produce")
    audio_pipeline.add_step(step)

    # Adding output step to the pipeline
    audio_pipeline.add_output(name="output", data_reference="steps.5.produce")

    return (audio_pipeline, tune_steps)
