from typing import Optional

from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from distil.primitives.load_edgelist import DistilEdgeListLoaderPrimitive
from distil.primitives.load_single_graph import DistilSingleGraphLoaderPrimitive
from distil.primitives.vertex_nomination import DistilVertexNominationPrimitive


def create_pipeline(
    metric: str, resolver: Optional[Resolver] = None, is_edgelist=False, min_meta=False
) -> Pipeline:
    # create the basic pipeline
    vertex_nomination_pipeline = Pipeline()
    vertex_nomination_pipeline.add_input(name="inputs")
    tune_steps = []

    # step 0 - extract the graphs
    if is_edgelist:
        step = PrimitiveStep(
            primitive_description=DistilEdgeListLoaderPrimitive.metadata.query(),
            resolver=resolver,
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="inputs.0",
        )
        step.add_output("produce")
        step.add_output("produce_target")
        vertex_nomination_pipeline.add_step(step)
    else:
        step = PrimitiveStep(
            primitive_description=DistilSingleGraphLoaderPrimitive.metadata.query(),
            resolver=resolver,
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="inputs.0",
        )
        step.add_output("produce")
        step.add_output("produce_target")
        vertex_nomination_pipeline.add_step(step)

    # step 1 - nominate
    step = PrimitiveStep(
        primitive_description=DistilVertexNominationPrimitive.metadata.query(),
        resolver=resolver,
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="steps.0.produce",
    )
    step.add_argument(
        name="outputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="steps.0.produce_target",
    )
    step.add_hyperparameter("metric", ArgumentType.VALUE, metric)
    step.add_output("produce")
    vertex_nomination_pipeline.add_step(step)
    tune_steps.append(1)

    # Adding output step to the pipeline
    vertex_nomination_pipeline.add_output(
        name="output", data_reference="steps.1.produce"
    )

    return (vertex_nomination_pipeline, tune_steps)
