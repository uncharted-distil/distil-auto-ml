from typing import Optional
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.metadata.base import ArgumentType

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.denormalize import DenormalizePrimitive
from d3m.primitives.object_detection import retinanet as RetinanetPrimitive


def create_pipeline(metric: str,
                    cat_mode: str = 'one_hot',
                    max_one_hot: int = 16,
                    scale: bool = False,
                    n_steps: int = 50,
                    resolver: Optional[Resolver] = None) -> Pipeline:
    # create the basic pipeline
    objdetect_pipeline = Pipeline()
    objdetect_pipeline.add_input(name='inputs')
    tune_steps = []

    # step 0 - denormalize dataframe (N.B.: injects semantic type information)
    step = PrimitiveStep(primitive_description=DenormalizePrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    # stepadd_hyperparameter('starting_resource', ArgumentType.VALUE, None)
    # step.add_hyperparameter('recursive', ArgumentType.VALUE, True)
    # step.add_hyperparameter('many_to_many', ArgumentType.VALUE, False)
    # step.add_hyperparameter('discard_not_joined_tabular_resources', ArgumentType.VALUE, False)
    objdetect_pipeline.add_step(step)

    # step 1 - extract dataframe from dataset
    step = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_output('produce')
    objdetect_pipeline.add_step(step)

    # step 4 - extract objects
    step = PrimitiveStep(primitive_description=RetinanetPrimitive.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step.add_hyperparameter('n_epochs', ArgumentType.VALUE, 30)
    step.add_hyperparameter('n_steps', ArgumentType.VALUE, n_steps)
    step.add_hyperparameter('batch_size', ArgumentType.VALUE, 8)
    step.add_output('produce')
    objdetect_pipeline.add_step(step)
    # tune_steps.append(2)

    # Adding output step to the pipeline
    objdetect_pipeline.add_output(name='output', data_reference='steps.2.produce')

    return (objdetect_pipeline, [])
