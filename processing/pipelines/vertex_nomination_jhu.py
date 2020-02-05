from typing import Optional
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m.metadata.base import ArgumentType

from distil.primitives.load_single_graph import DistilSingleGraphLoaderPrimitive
from distil.primitives.vertex_nomination import DistilVertexNominationPrimitive
from distil.primitives.load_edgelist import DistilEdgeListLoaderPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from d3m.primitives.data_transformation.load_graphs import JHU as JHUGraphLoader
from d3m.primitives.data_preprocessing.largest_connected_component import JHU as LCC
from d3m.primitives.data_transformation.adjacency_spectral_embedding import JHU as ASE
from d3m.primitives.classification.gaussian_classification import JHU as GCLASS

def create_pipeline(metric: str, resolver: Optional[Resolver] = None, is_edgelist=False, min_meta=False) -> Pipeline:
    # create the basic pipeline
    input_val = 'steps.{}.produce'
    vertex_nomination_pipeline = Pipeline()
    vertex_nomination_pipeline.add_input(name='inputs')

    # step 0 - extract the graphs
    step = PrimitiveStep(primitive_description=JHUGraphLoader.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    vertex_nomination_pipeline.add_step(step)
    previous_step = 0

    step = PrimitiveStep(primitive_description=LCC.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    vertex_nomination_pipeline.add_step(step)
    previous_step += 1

    step = PrimitiveStep(primitive_description=ASE.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    vertex_nomination_pipeline.add_step(step)
    previous_step += 1

    step = PrimitiveStep(primitive_description=GCLASS.metadata.query(), resolver=resolver)
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=input_val.format(previous_step))
    step.add_output('produce')
    vertex_nomination_pipeline.add_step(step)
    previous_step += 1

    # Adding output step to the pipeline
    vertex_nomination_pipeline.add_output(name='output', data_reference=input_val.format(previous_step))

    return vertex_nomination_pipeline
