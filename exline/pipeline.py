import json
import logging
from typing import Tuple

from d3m.container import dataset
from d3m import container, exceptions, runtime
from d3m.metadata import base as metadata_base, pipeline

from exline import router
from exline.modeling import metrics
from exline.pipelines import tabular

logger = logging.getLogger(__name__)

def create(dataset_doc_path: str, problem: dict, prepend: pipeline.Pipeline=None) -> Tuple[pipeline.Pipeline, container.Dataset]:
     # Load dataset in the same way the d3m runtime will
    train_dataset = dataset.Dataset.load(dataset_doc_path)

    # Load the dataset doc itself
    modified_path = dataset_doc_path.replace("file://", "")
    with open(modified_path) as json_file:
        dataset_doc = json.load(json_file)

    # extract metric from the problem
    protobuf_metric = problem['problem']['performanceMetrics'][0]['metric']
    metric = metrics.translate_proto_metric(protobuf_metric)

    # determine type of pipeline required for dataset
    pipeline_type, pipeline_info = router.get_routing_info(dataset_doc, problem, metric)

    if pipeline_type is 'table':
        pipeline = tabular.create_pipeline(metric)

    # prepend to the base pipeline
    if prepend is not None:
        pipeline = _prepend_pipeline(pipeline, prepend)

    return pipeline, train_dataset


def fit(pipeline: pipeline.Pipeline, problem: dict, input_dataset: container.Dataset) -> runtime.Runtime:
    hyperparams = None
    random_seed = 0
    volumes_dir = None

    fitted_pipeline, _, pipeline_run = runtime.fit(
        pipeline, problem, [input_dataset], hyperparams=hyperparams, random_seed=random_seed,
        volumes_dir=volumes_dir, context=metadata_base.Context.TESTING
    )

    return fitted_pipeline, pipeline_run


def produce(fitted_pipeline: runtime.Runtime, input_dataset: container.Dataset) -> container.DataFrame:
    results, _ = runtime.produce(fitted_pipeline, [input_dataset])
    return results


def _prepend_pipeline(base: pipeline.Pipeline, prepend: pipeline.Pipeline) -> pipeline.Pipeline:
    # wrap pipeline in a sub pipeline - d3m core node replacement function doesn't work otherwise
    subpipeline = pipeline.SubpipelineStep(pipeline=base)

    # find the placeholder node in the prepend and replace it with the base sub pipeline
    for i, step in enumerate(prepend.steps):
        if isinstance(step, pipeline.PlaceholderStep):
            # set inputs/outputs manually since the replace doesn't infer them
            for input_ref in step.get_input_data_references():
                subpipeline.add_input(input_ref)
            for output_id in step.outputs:
                subpipeline.add_output(output_id)

            prepend.replace_step(i, subpipeline)
            return prepend

    logger.warn(f'Failed to prepend pipeline {prepend.id} - continuing with base unmodified')
    return base
