from d3m import index

from typing import Optional

from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from d3m.primitives.data_preprocessing.data_cleaning import DistilTimeSeriesFormatter
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
from d3m.primitives.time_series_classification.convolutional_neural_net import LSTM_FCN
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.simple_profiler import SimpleProfilerPrimitive

def create_pipeline(metric: str, min_meta: bool = False, resolver: Optional[Resolver] = None) -> Pipeline:
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')
    input_val = 'steps.{}.produce'
    tune_steps = []
    # Step 1: Ts formatter
    step_0 = PrimitiveStep(primitive_description=DistilTimeSeriesFormatter.metadata.query(),
        resolver=resolver)
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)
    previous_step = 0

    # Step 2: DS to DF on formatted ts DS
    step_1 = PrimitiveStep(DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver)
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)
    previous_step += 1
    ts_parse_step = previous_step


    # Step 3: DS to DF on input DS
    step_2 = PrimitiveStep(DatasetToDataFramePrimitive.metadata.query(),
        resolver=resolver)
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_2.add_output('produce')
    pipeline_description.add_step(step_2)
    previous_step += 1
    parse_step = previous_step

    if min_meta:
        step = PrimitiveStep(primitive_description=SimpleProfilerPrimitive.metadata.query(), resolver=resolver)
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                          data_reference=input_val.format(parse_step))
        # step.add_hyperparameter(name='overwrite', argument_type=ArgumentType.VALUE, data=True)
        step.add_output('produce')
        pipeline_description.add_step(step)
        previous_step += 1
        parse_step = previous_step
        tune_steps.append(previous_step)

        # # Extract attributes
        # step = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
        #                      resolver=resolver)
        # step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
        #                   data_reference=input_val.format(parse_step))
        # step.add_output('produce')
        # step.add_hyperparameter('semantic_types', ArgumentType.VALUE,
        #                         ('https://metadata.datadrivendiscovery.org/types/Attribute',))
        # step.add_hyperparameter('negate', ArgumentType.VALUE, True)
        # step.add_hyperparameter('match_logic', ArgumentType.VALUE, 'equal')
        # pipeline_description.add_step(step)
        # previous_step += 1
        # parse_step = previous_step

        step = PrimitiveStep(primitive_description=SimpleProfilerPrimitive.metadata.query(), resolver=resolver)
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                          data_reference=input_val.format(ts_parse_step))
        # step.add_hyperparameter(name='overwrite', argument_type=ArgumentType.VALUE, data=True)
        step.add_output('produce')
        pipeline_description.add_step(step)
        previous_step += 1
        ts_parse_step = previous_step
        tune_steps.append(previous_step)


    # step 3: column parser on input DF
    step_3 = PrimitiveStep(ColumnParserPrimitive.metadata.query())
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step_3.add_output('produce')
    step_3.add_hyperparameter(name='parse_semantic_types', argument_type=ArgumentType.VALUE,
                              data=["http://schema.org/Boolean",
                                    "http://schema.org/Integer",
                                    "http://schema.org/Float",
                                    "https://metadata.datadrivendiscovery.org/types/FloatVector"])
    pipeline_description.add_step(step_3)
    previous_step += 1

    # Step 5: parse target semantic types
    step_4 = PrimitiveStep(ExtractColumnsBySemanticTypesPrimitive.metadata.query(),
        resolver=resolver)
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step_4.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                              data=["https://metadata.datadrivendiscovery.org/types/Target",
                                    "https://metadata.datadrivendiscovery.org/types/TrueTarget"])
    step_4.add_output('produce')
    pipeline_description.add_step(step_4)
    previous_step += 1

    # Step 6: LSTM_FCN
    step_5 = PrimitiveStep(LSTM_FCN.metadata.query(),
        resolver=resolver)
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(ts_parse_step))
    step_5.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)
    previous_step += 1
    tune_steps.append(previous_step)

    # Step 7: construct predictions
    step_6 = PrimitiveStep(ConstructPredictionsPrimitive.metadata.query(),
        resolver=resolver)
    step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step_6.add_output('produce')
    pipeline_description.add_step(step_6)
    previous_step += 1

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference=input_val.format(previous_step))

    return (pipeline_description, tune_steps)
