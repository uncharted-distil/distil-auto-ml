from typing import Optional
import numpy
import config
import os
import uuid

from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from common_primitives.extract_columns_structural_types import (
    ExtractColumnsByStructuralTypesPrimitive,
)
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.primitives.remote_sensing.mlp import MlpClassifier
from distil.primitives.ensemble_forest import EnsembleForestPrimitive
from distil.primitives.ranked_linear_svc import RankedLinearSVCPrimitive
from distil.primitives.column_parser import ColumnParserPrimitive
from distil.primitives.enrich_dates import EnrichDatesPrimitive
from distil.primitives.list_to_dataframe import ListEncoderPrimitive
from distil.primitives.prefeaturised_pooler import PrefeaturisedPoolingPrimitive
from dsbox.datapreprocessing.cleaner.iterative_regression import (
    IterativeRegressionImputation,
)


def create_pipeline(
    metric: str,
    profiler="none",
    predictive_primitive="svc",
    is_pooled=True,
    batch_size: int = 128,
    n_jobs=-1,
    resolver: Optional[Resolver] = None,
) -> Pipeline:
    input_val = "steps.{}.produce"
    tune_steps = []

    # create the basic pipeline
    rs_pretrained_pipeline = Pipeline()
    rs_pretrained_pipeline.add_input(name="inputs")

    # extract dataframe from dataset
    step = PrimitiveStep(
        primitive_description=DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step.add_output("produce")
    rs_pretrained_pipeline.add_step(step)
    previous_step = 0

    if profiler == "simple":
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
        rs_pretrained_pipeline.add_step(step)
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
        "http://schema.org/Integer",
        "http://schema.org/Float",
        "https://metadata.datadrivendiscovery.org/types/FloatVector",
    )
    step.add_hyperparameter("parsing_semantics", ArgumentType.VALUE, semantic_types)
    rs_pretrained_pipeline.add_step(step)
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
        ("https://metadata.datadrivendiscovery.org/types/Attribute",),
    )
    rs_pretrained_pipeline.add_step(step)
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
    rs_pretrained_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    if not is_pooled and predictive_primitive != "mlp":
        step = PrimitiveStep(
            primitive_description=PrefeaturisedPoolingPrimitive.metadata.query(),
            resolver=resolver,
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(attributes_step),
        )
        step.add_output("produce")
        step.add_hyperparameter("batch_size", ArgumentType.VALUE, batch_size)
        rs_pretrained_pipeline.add_step(step)
        previous_step += 1
        attributes_step = previous_step

    if predictive_primitive != "mlp":
        # List encoder to get from vectors to columns
        step = PrimitiveStep(
            primitive_description=ListEncoderPrimitive.metadata.query(),
            resolver=resolver,
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(attributes_step),
        )
        step.add_output("produce")
        rs_pretrained_pipeline.add_step(step)
        previous_step += 1

        # Enrich any dates present
        step = PrimitiveStep(
            primitive_description=EnrichDatesPrimitive.metadata.query(),
            resolver=resolver,
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(previous_step),
        )
        step.add_output("produce")
        rs_pretrained_pipeline.add_step(step)
        previous_step += 1
        attributes_step = previous_step

    # # Extract floats to ensure that we're only passing valid data into the learner
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
        ("float", "numpy.float32", "numpy.float64"),
    )
    rs_pretrained_pipeline.add_step(step)
    previous_step += 1

    # Impute any missing data
    step = PrimitiveStep(
        primitive_description=IterativeRegressionImputation.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    rs_pretrained_pipeline.add_step(step)
    previous_step += 1

    if predictive_primitive == "svc":
        # Generates a linear svc model.
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
        step.add_hyperparameter("scaling", ArgumentType.VALUE, "unit_norm")
        step.add_hyperparameter("rank_confidences", ArgumentType.VALUE, True)
        step.add_hyperparameter("calibrate", ArgumentType.VALUE, True)
        step.add_output("produce")
        rs_pretrained_pipeline.add_step(step)
        previous_step += 1
        tune_steps.append(previous_step)
    elif predictive_primitive == "forest":
        # Generates a random forest ensemble model.
        step = PrimitiveStep(
            primitive_description=EnsembleForestPrimitive.metadata.query(),
            resolver=resolver,
        )
        step.add_hyperparameter("grid_search", ArgumentType.VALUE, False)
        step.add_hyperparameter("small_dataset_fits", ArgumentType.VALUE, 1)
        step.add_hyperparameter("metric", ArgumentType.VALUE, metric)
        step.add_hyperparameter("n_jobs", ArgumentType.VALUE, n_jobs)
        step.add_hyperparameter("compute_confidences", ArgumentType.VALUE, True)
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
        rs_pretrained_pipeline.add_step(step)
        previous_step += 1
        tune_steps.append(previous_step)
    elif predictive_primitive == "mlp" and not is_pooled:
        step = PrimitiveStep(
            primitive_description=MlpClassifier.metadata.query(), resolver=resolver
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
        step.add_hyperparameter("all_confidences", ArgumentType.VALUE, True)
        step.add_hyperparameter(
            "weights_filepath",
            ArgumentType.VALUE,
            os.path.join(
                config.OUTPUT_DIR, "mlp_classifier_" + str(uuid.uuid4()) + ".pth"
            ),
        )
        # always return a 4x4 explanation matrix
        step.add_hyperparameter("image_dim", ArgumentType.VALUE, 4)
        step.add_output("produce")
        rs_pretrained_pipeline.add_step(step)
        previous_step += 1
        tune_steps.append(previous_step)

    # # convert predictions to expected format
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
    rs_pretrained_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    rs_pretrained_pipeline.add_output(
        name="output", data_reference=input_val.format(previous_step)
    )

    return (rs_pretrained_pipeline, tune_steps)
