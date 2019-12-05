from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from typing import Optional
import sys

def create_pipeline(metric: str, resolver: Optional[Resolver] = None) -> Pipeline:
    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')

    # Step 0: Denormalize primitive
    step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.denormalize.Common'), resolver=resolver)
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 1, 2 column parser
    step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.DataFrameCommon'), resolver=resolver)
    pipeline_description.add_step(step_1)

    step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon'), resolver=resolver)
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_2.add_hyperparameter(name='primitive', argument_type= ArgumentType.PRIMITIVE, data=1)
    step_2.add_output('produce')
    pipeline_description.add_step(step_2)

    # Step 3,4: imputer
    step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'), resolver=resolver)
    step_3.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE,data='replace')
    step_3.add_hyperparameter(name='use_semantic_types', argument_type=ArgumentType.VALUE,data=True)
    pipeline_description.add_step(step_3)

    step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon'), resolver=resolver)
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_4.add_hyperparameter(name='primitive', argument_type= ArgumentType.PRIMITIVE, data=3)
    step_4.add_output('produce')
    pipeline_description.add_step(step_4)

    # Step 5: DISTIL/NK Storc primitive
    step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.clustering.hdbscan.Hdbscan'), resolver=resolver)
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    step_5.add_hyperparameter(name='long_format', argument_type= ArgumentType.VALUE, data=True)
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)

    # Step 6,7,8: Distil ensemble classifier
    step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'), resolver=resolver)
    step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
    step_6.add_output('produce')
    pipeline_description.add_step(step_6)

    step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'), resolver=resolver)
    step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
    step_7.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,data=('https://metadata.datadrivendiscovery.org/types/Target',))
    step_7.add_output('produce')
    pipeline_description.add_step(step_7)

    step_8 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.learner.random_forest.DistilEnsembleForest'), resolver=resolver)
    step_8.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
    step_8.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.7.produce')
    step_8.add_output('produce')
    pipeline_description.add_step(step_8)

    # Step 9, 10: construct output
    step_9 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.DataFrameCommon'), resolver=resolver)
    step_9.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.8.produce')
    step_9.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
    step_9.add_output('produce')
    pipeline_description.add_step(step_9)

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference='steps.9.produce')
    return pipeline_description
