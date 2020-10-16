from typing import Optional

from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.add_semantic_types import AddSemanticTypesPrimitive
from distil.primitives.mi_ranking import MIRankingPrimitive
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver


def create_pipeline(metric: str,
                    resolver: Optional[Resolver] = None) -> Pipeline:
    input_val = 'steps.{}.produce'
    tune_steps = []

    mi_ranking_pipeline = Pipeline()
    mi_ranking_pipeline.add_input(name="inputs")

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
        primitive_description=AddSemanticTypesPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step)
    )
    step.add_output("produce")
    step.add_hyperparameter("columns", ArgumentType.VALUE, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12])
    step.add_hyperparameter("semantic_types", ArgumentType.VALUE, ("http://schema.org/Integer",))
    mi_ranking_pipeline.add_step(step)
    previous_step += 1

    step = PrimitiveStep(
        primitive_description=AddSemanticTypesPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step)
    )
    step.add_output("produce")
    step.add_hyperparameter("columns", ArgumentType.VALUE, [13, 14, 15, 16])
    step.add_hyperparameter("semantic_types", ArgumentType.VALUE, ("http://schema.org/Float",))
    mi_ranking_pipeline.add_step(step)
    previous_step += 1

    # step = PrimitiveStep(
    #     primitive_description=AddSemanticTypesPrimitive.metadata.query(), resolver=resolver
    # )
    # step.add_argument(
    #     name="inputs",
    #     argument_type=ArgumentType.CONTAINER,
    #     data_reference=input_val.format(previous_step)
    # )
    # step.add_output("produce")
    # step.add_hyperparameter("columns", ArgumentType.VALUE, [17])
    # step.add_hyperparameter("semantic_types", ArgumentType.VALUE, ("http://schema.org/Text",))
    # mi_ranking_pipeline.add_step(step)
    # previous_step += 1

    # step = PrimitiveStep(
    #     primitive_description=AddSemanticTypesPrimitive.metadata.query(), resolver=resolver
    # )
    # step.add_argument(
    #     name="inputs",
    #     argument_type=ArgumentType.CONTAINER,
    #     data_reference=input_val.format(previous_step)
    # )
    # step.add_output("produce")
    # step.add_hyperparameter("columns", ArgumentType.VALUE, [17])
    # step.add_hyperparameter("semantic_types", ArgumentType.VALUE, ("https://metadata.datadrivendiscovery.org/types/CategoricalData",))
    # mi_ranking_pipeline.add_step(step)
    # previous_step += 1

    step = PrimitiveStep(
        primitive_description=AddSemanticTypesPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step)
    )
    step.add_output("produce")
    step.add_hyperparameter("columns", ArgumentType.VALUE, [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    step.add_hyperparameter("semantic_types", ArgumentType.VALUE, ("https://metadata.datadrivendiscovery.org/types/Attribute",))
    mi_ranking_pipeline.add_step(step)
    previous_step += 1

    step = PrimitiveStep(
        primitive_description=AddSemanticTypesPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step)
    )
    step.add_output("produce")
    step.add_hyperparameter("columns", ArgumentType.VALUE, [6])
    step.add_hyperparameter("semantic_types", ArgumentType.VALUE, ("https://metadata.datadrivendiscovery.org/types/Target",))
    mi_ranking_pipeline.add_step(step)
    previous_step += 1

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
    step.add_hyperparameter("parse_semantic_types", ArgumentType.VALUE, semantic_types)
    mi_ranking_pipeline.add_step(step)
    previous_step += 1
    parse_step = previous_step

    step = PrimitiveStep(
        primitive_description=MIRankingPrimitive.metadata.query(),
        resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference=input_val.format(previous_step)
    )
    step.add_output("produce")
    step.add_hyperparameter(
        "target_col_index",
        ArgumentType.VALUE,
        6
    )
    mi_ranking_pipeline.add_step(step)
    previous_step += 1

    mi_ranking_pipeline.add_output(
        name="output", data_reference=input_val.format(previous_step)
    )

    return (mi_ranking_pipeline, tune_steps)
