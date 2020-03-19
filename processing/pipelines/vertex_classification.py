from typing import Optional
from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.metadata.base import ArgumentType

from sri.graph.vertex_classification import VertexClassificationParser
from sri.psl.vertex_classification import VertexClassification


def create_pipeline(metric: str, resolver: Optional[Resolver] = None) -> Pipeline:
    # create the basic pipeline
    vertex_classification_pipeline = Pipeline()
    vertex_classification_pipeline.add_input(name='inputs')

    # step 0 - extract the graphs
    step = PrimitiveStep(primitive_description=VertexClassificationParser.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    vertex_classification_pipeline.add_step(step)

    # step 1 - classify
    step = PrimitiveStep(primitive_description=VertexClassification.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step.add_hyperparameter('jvm_memory', ArgumentType.VALUE, 0.6)
    step.add_output('produce')
    vertex_classification_pipeline.add_step(step)

    # Adding output step to the pipeline
    vertex_classification_pipeline.add_output(name='output', data_reference='steps.1.produce')

    return (vertex_classification_pipeline, [])
