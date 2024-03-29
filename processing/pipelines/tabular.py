from typing import Optional

from distil.primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from common_primitives.xgboost_gbtree import XGBoostGBTreeClassifierPrimitive
from common_primitives.xgboost_regressor import XGBoostGBTreeRegressorPrimitive
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.primitives.clustering.hdbscan import Hdbscan
from d3m.primitives.data_cleaning.column_type_profiler import Simon
from processing.metrics import (
    regression_metrics,
    classification_metrics,
    confidence_metrics,
)
from distil.primitives.binary_encoder import BinaryEncoderPrimitive
from distil.primitives.categorical_imputer import CategoricalImputerPrimitive
from distil.primitives.enrich_dates import EnrichDatesPrimitive
from distil.primitives.ensemble_forest import EnsembleForestPrimitive
from distil.primitives.list_to_dataframe import ListEncoderPrimitive
from distil.primitives.one_hot_encoder import OneHotEncoderPrimitive
from distil.primitives.replace_singletons import ReplaceSingletonsPrimitive
from distil.primitives.text_encoder import TextEncoderPrimitive
from sklearn_wrap import SKMissingIndicator, SKImputer
from sklearn_wrap import SKStandardScaler
from dsbox.datapreprocessing.cleaner.iterative_regression import (
    IterativeRegressionImputation,
)


def create_pipeline(
    metric: str,
    semi: bool = False,
    cat_mode: str = "one_hot",
    max_one_hot: int = 16,
    scale: bool = False,
    include_one_hot=True,
    profiler="none",
    use_boost: bool = True,
    grid_search=False,
    n_jobs: int = -1,
    compute_confidences=True,
    pos_label=None,
    resolver: Optional[Resolver] = None,
) -> Pipeline:
    input_val = "steps.{}.produce"
    tune_steps = []

    if not include_one_hot:
        max_one_hot = 0

    # create the basic pipeline
    tabular_pipeline = Pipeline()
    tabular_pipeline.add_input(name="inputs")

    # extract dataframe from dataset
    step = PrimitiveStep(
        primitive_description=DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step.add_output("produce")
    tabular_pipeline.add_step(step)
    previous_step = 0

    if profiler == "simon":
        step = PrimitiveStep(
            primitive_description=Simon.metadata.query(), resolver=resolver
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(previous_step),
        )
        step.add_hyperparameter(
            name="overwrite", argument_type=ArgumentType.VALUE, data=True
        )
        step.add_hyperparameter(
            name="multi_label_classification",
            argument_type=ArgumentType.VALUE,
            data=False,
        )
        step.add_output("produce")
        tabular_pipeline.add_step(step)
        previous_step += 1
    elif profiler == "simple":
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
        tabular_pipeline.add_step(step)
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
    tabular_pipeline.add_step(step)
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
    tabular_pipeline.add_step(step)
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
    tabular_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    # Append date enricher.  Looks for date columns and normalizes them.
    step = PrimitiveStep(
        primitive_description=EnrichDatesPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(attributes_step),
    )
    step.add_output("produce")
    tabular_pipeline.add_step(step)
    previous_step += 1

    # append list encoder. Take list and expand them across columns
    step = PrimitiveStep(
        primitive_description=ListEncoderPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(attributes_step),
    )
    step.add_output("produce")
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Append singleton replacer.  Looks for categorical values that only occur once in a column and replace them with a flag.
    step = PrimitiveStep(
        primitive_description=ReplaceSingletonsPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Append categorical imputer.  Finds missing categorical values and replaces them with an imputed value.
    step = PrimitiveStep(
        primitive_description=CategoricalImputerPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Adds an svm text encoder for text fields.
    step = PrimitiveStep(
        primitive_description=TextEncoderPrimitive.metadata.query(), resolver=resolver
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
    if metric in regression_metrics:
        step.add_hyperparameter("encoder_type", ArgumentType.VALUE, "tfidf")
    step.add_output("produce")
    tabular_pipeline.add_step(step)
    previous_step += 1
    tune_steps.append(previous_step)

    # Adds a one hot encoder for categoricals of low cardinality.
    if cat_mode == "one_hot" and include_one_hot:
        step = PrimitiveStep(
            primitive_description=OneHotEncoderPrimitive.metadata.query(),
            resolver=resolver,
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(previous_step),
        )
        step.add_output("produce")
        step.add_hyperparameter("max_one_hot", ArgumentType.VALUE, max_one_hot)
        tabular_pipeline.add_step(step)
        previous_step += 1

    # Adds a binary encoder for categoricals of high cardinality.
    step = PrimitiveStep(
        primitive_description=BinaryEncoderPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step),
    )
    step.add_output("produce")
    if cat_mode == "one_hot":
        step.add_hyperparameter("min_binary", ArgumentType.VALUE, max_one_hot + 1)
    else:
        step.add_hyperparameter("min_binary", ArgumentType.VALUE, 0)
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Adds SK learn missing value indicator (list encoder must be run first to prevent the above failure)
    # Currently disabled due to the fact that it will only append a missing indicator column on produce if there is
    # missing data present.  This leads to issues when there is missing data in the fit set and not the produce set, as the
    # training features don't match the produce features.
    #

    #
    #
    # Adds SK learn simple imputer
    # step = PrimitiveStep(primitive_description=SKImputer.SKImputer.metadata.query())
    # use dsbox imputer since sklearn has issues with mismatch datasets
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
    # step.add_hyperparameter('use_semantic_types', ArgumentType.VALUE, True)
    # step.add_hyperparameter('error_on_no_input', ArgumentType.VALUE, False)
    # step.add_hyperparameter('return_result', ArgumentType.VALUE, 'replace')
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Append scaler for numerical values.
    if scale:
        step = PrimitiveStep(
            primitive_description=SKStandardScaler.SKStandardScaler.metadata.query(),
            resolver=resolver,
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(previous_step),
        )
        step.add_output("produce")
        step.add_hyperparameter("use_semantic_types", ArgumentType.VALUE, True)
        step.add_hyperparameter("return_result", ArgumentType.VALUE, "replace")
        tabular_pipeline.add_step(step)
        previous_step += 1

    if semi:
        step = PrimitiveStep(
            primitive_description=SKMissingIndicator.SKMissingIndicator.metadata.query(),
            resolver=resolver,
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(previous_step),
        )
        step.add_output("produce")
        step.add_hyperparameter("use_semantic_types", ArgumentType.VALUE, True)
        step.add_hyperparameter("return_result", ArgumentType.VALUE, "append")
        step.add_hyperparameter("error_on_new", ArgumentType.VALUE, False)
        step.add_hyperparameter("error_on_no_input", ArgumentType.VALUE, False)
        tabular_pipeline.add_step(step)
        previous_step += 1

        # hdb scan to add clustering features.
        step = PrimitiveStep(
            primitive_description=Hdbscan.metadata.query(), resolver=resolver
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=input_val.format(previous_step),
        )
        step.add_hyperparameter("required_output", ArgumentType.VALUE, "feature")
        step.add_output("produce")
        tabular_pipeline.add_step(step)
        previous_step += 1

    # Generates a random forest ensemble model.
    # step = PrimitiveStep(primitive_description=EnsembleForestPrimitive.metadata.query(), resolver=resolver)
    if use_boost:
        if metric in regression_metrics:
            step = PrimitiveStep(
                primitive_description=XGBoostGBTreeRegressorPrimitive.metadata.query(),
                resolver=resolver,
            )
            step.add_hyperparameter("n_jobs", ArgumentType.VALUE, n_jobs)
        elif metric in classification_metrics and metric not in confidence_metrics:
            # xgboost classifier doesn't support probability generation so no support for confidence-based metrics
            step = PrimitiveStep(
                primitive_description=XGBoostGBTreeClassifierPrimitive.metadata.query(),
                resolver=resolver,
            )
            step.add_hyperparameter("n_jobs", ArgumentType.VALUE, n_jobs)
        else:
            return None
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
        step.add_hyperparameter("pos_label", ArgumentType.VALUE, pos_label)
        step.add_hyperparameter("n_jobs", ArgumentType.VALUE, n_jobs)
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
    if not use_boost:
        step.add_hyperparameter("metric", ArgumentType.VALUE, metric)

    tabular_pipeline.add_step(step)
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
    tabular_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    tabular_pipeline.add_output(
        name="output", data_reference=input_val.format(previous_step)
    )

    return (tabular_pipeline, tune_steps)
