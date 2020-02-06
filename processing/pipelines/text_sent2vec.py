from typing import Optional
from d3m.metadata.pipeline import Resolver
from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
import logging

logging.basicConfig(level=logging.DEBUG)


# CDB: Totally unoptimized.  Pipeline creation code could be simplified but has been left
# in a naively implemented state for readability for now.
#
# Overall implementation relies on passing the entire dataset through the pipeline, with the primitives
# identifying columns to operate on based on type.  Alternative implementation (that better lines up with
# D3M approach, but generates more complex pipelines) would be to extract sub-sets by semantic type using
# a common primitive, apply the type-specific primitive to the sub-set, and then merge the changes
# (replace or join) back into the original data.
def create_pipeline(metric: str,
                    cat_mode: str = 'one_hot',
                    max_one_hot: int = 16,
                    scale: bool = False,
                    min_meta: bool = False,
                    resolver: Optional[Resolver] = None) -> Pipeline:
    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name="inputs")

    # Step 0: Denormalize primitive
    step_0 = PrimitiveStep(
        primitive=index.get_primitive(
            "d3m.primitives.data_transformation.denormalize.Common"
        )
    )
    step_0.add_argument(
        name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
    )
    step_0.add_output("produce")
    pipeline_description.add_step(step_0)

    # Step 1: dataset_to_dataframe
    step_1 = PrimitiveStep(
        primitive=index.get_primitive(
            "d3m.primitives.data_transformation.dataset_to_dataframe.Common"
        )
    )
    step_1.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="steps.0.produce",
    )
    step_1.add_hyperparameter(
        name="dataframe_resource", argument_type=ArgumentType.VALUE, data="learningData"
    )
    step_1.add_output("produce")
    pipeline_description.add_step(step_1)

    # Step 2: sent2vec_wrapper primitive
    step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec'))
    step_2.add_argument(
        name="inputs",
        argument_type=ArgumentType.CONTAINER,
        data_reference="steps.1.produce",
    )
    step_2.add_output("produce")
    pipeline_description.add_step(step_2)

    # Step 3: column_parser
    step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'))
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_output('produce')
    pipeline_description.add_step(step_3)

    # # Step 4: imputer
    # step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
    # step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    # step_3.add_output('produce')
    # step_3.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE,data='replace')
    # step_3.add_hyperparameter(name='use_semantic_types', argument_type=ArgumentType.VALUE,data=True)
    # pipeline_description.add_step(step_3)

    # Step 4: Gradient boosting classifier
    step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.classification.xgboost_gbtree.Common'))
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_output('produce')
    step_4.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE, data='replace')
    pipeline_description.add_step(step_4)

    # Step 5: construct output
    step_5 = PrimitiveStep(
        primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    step_5.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference='steps.5.produce')

    return pipeline_description
