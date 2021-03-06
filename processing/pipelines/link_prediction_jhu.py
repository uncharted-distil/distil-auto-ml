from typing import Optional

from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.primitives.link_prediction.data_conversion import JHU as JHULinkLoader
from d3m.primitives.link_prediction.rank_classification import JHU as JHURankPrimitive


# from d3m.primitives.data_transformation.load_graphs import JHU as JHUGraphLoader
def create_pipeline(metric: str, resolver: Optional[Resolver] = None) -> Pipeline:

    # create the basic pipeline
    vertex_nomination_pipeline = Pipeline()
    vertex_nomination_pipeline.add_input(name="inputs")

    # step 0 - extract the graphs
    step = PrimitiveStep(
        primitive_description=JHULinkLoader.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step.add_output("produce")
    vertex_nomination_pipeline.add_step(step)

    # step 1 - predict links
    step = PrimitiveStep(
        primitive_description=JHURankPrimitive.metadata.query(), resolver=resolver
    )
    step.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="steps.0.produce",
    )
    step.add_output("produce")
    vertex_nomination_pipeline.add_step(step)

    # Adding output step to the pipeline
    vertex_nomination_pipeline.add_output(
        name="output", data_reference="steps.1.produce"
    )

    return (vertex_nomination_pipeline, [])
