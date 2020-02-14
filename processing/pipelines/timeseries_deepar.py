from typing import Optional

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.metadata.base import ArgumentType

from d3m.primitives.time_series_forecasting.lstm import DeepAR
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.grouping_field_compose import GroupingFieldComposePrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from d3m.primitives.data_transformation import construct_predictions
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
from common_primitives.simple_profiler import SimpleProfilerPrimitive

PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)


def create_pipeline(metric: str, resolver: Optional[Resolver] = None) -> Pipeline:
    previous_step = 0
    input_val = 'steps.{}.produce'

    # create the basic pipeline
    deepar_pipeline = Pipeline(context=PipelineContext.TESTING)
    deepar_pipeline.add_input(name='inputs')

    # step 0 - Extract dataframe from dataset
    step = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    deepar_pipeline.add_step(step)


    step = PrimitiveStep(primitive_description=SimpleProfilerPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                      data_reference=input_val.format(previous_step))
    step.add_output('produce')
    deepar_pipeline.add_step(step)
    previous_step += 1

    # step 1 - Parse columns.
    step = PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                      data_reference=input_val.format(previous_step))
    step.add_output('produce')
    semantic_types = ("http://schema.org/Boolean",
                      "http://schema.org/Integer",
                      "http://schema.org/Float",
                      "https://metadata.datadrivendiscovery.org/types/FloatVector",
                      "http://schema.org/DateTime")
    step.add_hyperparameter('parse_semantic_types', ArgumentType.VALUE, semantic_types)
    deepar_pipeline.add_step(step)
    previous_step += 1
    parse_step = previous_step

    # Step 2: parse attribute semantic types
    step = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
                         resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                            data=["https://metadata.datadrivendiscovery.org/types/Attribute"])
    step.add_output('produce')
    deepar_pipeline.add_step(step)
    previous_step += 1
    attribute_step = previous_step

    # Step 3: Grouping Field Compose
    step = PrimitiveStep(
        primitive_description=GroupingFieldComposePrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                      data_reference=input_val.format(previous_step))
    step.add_output('produce')
    deepar_pipeline.add_step(step)
    previous_step += 1

    # Step 4: parse target semantic types
    step = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
                         resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                            data=["https://metadata.datadrivendiscovery.org/types/Target",
                                  "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                                  "https://metadata.datadrivendiscovery.org/types/SuggestedTarget"])
    step.add_output('produce')
    deepar_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    # step 5 - Forecasting Primitive
    step = PrimitiveStep(primitive_description=DeepAR.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                      data_reference=input_val.format(attribute_step))
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER,
                      data_reference=input_val.format(target_step))
    step.add_output('produce')
    deepar_pipeline.add_step(step)
    previous_step += 1

    # Step 6: construct predictions
    step = PrimitiveStep(
        primitive_description=construct_predictions.Common.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                      data_reference=input_val.format(previous_step))
    step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER,
                      data_reference=input_val.format(parse_step))
    step.add_output('produce')
    deepar_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    deepar_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return deepar_pipeline
