from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive

# from d3m.primitives.time_series_classification.k_neighbors import Kanine
# from distil.primitives.timeseries_formatter import TimeSeriesFormatterPrimitive
# from d3m.primitives.data_preprocessing.data_cleaning import DistilTimeSeriesFormatter
# from distil.primitives.column_grouper import ColumnGrouperPrimitive


def create_pipeline(metric: str) -> Pipeline:
    # create the basic pipeline
    kanine_pipeline = Pipeline()
    kanine_pipeline.add_input(name='inputs')

    # step 0 - flatten the timeseries if necessary
    step = PrimitiveStep(primitive_description=DistilTimeSeriesFormatter.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    kanine_pipeline.add_step(step)
    previous_step = 0
    parse_step = previous_step

    # extract dataframe from dataset
    step = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(parse_step))
    step.add_output('produce')
    kanine_pipeline.add_step(step)
    previous_step += 1
    df_step = previous_step

    # goup target column based on grouper key.
    step = PrimitiveStep(primitive_description=ColumnGrouperPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(df_step))
    step.add_output('produce')
    kanine_pipeline.add_step(step)
    previous_step += 1
    target_step = previous_step

    # step 1 - kanine classification
    step = PrimitiveStep(primitive_description=Kanine.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=df_step)
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=target_step)
    step.add_hyperparameter('long_format', ArgumentType.VALUE, True)
    step.add_output('produce')
    kanine_pipeline.add_step(step)

    # Adding output step to the pipeline
    kanine_pipeline.add_output(name='output', data_reference='steps.1.produce')

    return kanine_pipeline
