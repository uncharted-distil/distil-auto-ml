import json
import logging
import math
import os
import pickle
import time
import typing
import copy
from collections import defaultdict
from multiprocessing import Process
from typing import Tuple, List, Dict, Optional

import GPUtil
import config
import numpy as np
import pandas as pd
import sherpa
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.extract_columns_semantic_types import (
    ExtractColumnsBySemanticTypesPrimitive,
)
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from d3m import container, runtime
from d3m import primitives
from d3m.container import dataset
from d3m.metadata import (
    base as metadata_base,
    pipeline,
    problem,
    pipeline_run,
    hyperparams,
)
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import (
    Pipeline,
    PlaceholderStep,
    Resolver,
)
from d3m.metadata.problem import PerformanceMetricBase, PerformanceMetric
from processing import metrics
from processing import router
from processing.pipelines import (
    clustering,
    collaborative_filtering,
    graph_matching,
    image,
    remote_sensing,
    object_detection,
    object_detection_yolo,
    question_answer,
    tabular,
    text,
    text_sent2vec,
    link_prediction,
    link_prediction_jhu,
    audio,
    vertex_nomination,
    # vertex_nomination_jhu,
    vertex_classification,
    community_detection,
    timeseries_kanine,
    timeseries_var,
    timeseries_lstm_fcn,
    semisupervised_tabular,
    timeseries_deepar,
)
import pymongo
import signal
import psutil

# data_augmentation_tabular)

logger = logging.getLogger(__name__)


def create(
    dataset_doc_path: str,
    problem: dict,
    time_limit: int,
    prepend: Optional[Pipeline] = None,
    resolver: Optional[Resolver] = None,
) -> Tuple[List[pipeline.Pipeline], container.Dataset, List[float]]:
    # allow for use of GPU optimized pipelines
    gpu = _use_gpu()

    # Optionally enable external hyperparameter tuning.
    tune_pipeline = config.HYPERPARAMETER_TUNING

    # Load dataset in the same way the d3m runtime will
    train_dataset = dataset.Dataset.load(dataset_doc_path)

    # If there isn't a placeholder this is a fully specified pipeline.  Return the pipeline unmodified along with the
    # dataset.
    if prepend and not [True for s in prepend.steps if isinstance(s, PlaceholderStep)]:
        return ([prepend], train_dataset, [1.0])

    # Load the dataset doc itself
    modified_path = dataset_doc_path.replace("file://", "")
    with open(modified_path) as json_file:
        dataset_doc = json.load(json_file)

    # extract metric from the problem
    protobuf_metric = problem["problem"]["performance_metrics"][0]["metric"]
    metric = metrics.translate_proto_metric(protobuf_metric)
    logger.info(f"Optimizing on metric {metric}")

    # determine type of pipeline required for dataset
    pipeline_type, pipeline_info = router.get_routing_info(dataset_doc, problem, metric)

    # Check if all columns have valid metadata
    # TODO check for unknown types as well.
    learning_data_col = [
        x for x in dataset_doc["dataResources"] if x["resID"] == "learningData"
    ]
    num_of_resources = len(
        [x for x in learning_data_col[0]["columns"] if x.get("colType") is not None]
    )

    if num_of_resources < len(train_dataset["learningData"].columns):
        MIN_META = True
    else:
        MIN_META = False

    if prepend is not None:
        # if we have any prepend steps that modify semantic types, min_meta check no longer applies
        prepend_steps = {step.primitive for step in prepend.steps[:-1]}
        semantic_modifiers = {
            primitives.data_transformation.add_semantic_types.Common,
            primitives.data_transformation.remove_semantic_types.Common,
            primitives.data_cleaning.column_type_profiler.Simon,
            SimpleProfilerPrimitive,
        }
        if len(prepend_steps & semantic_modifiers) > 0:
            logger.info("Metadata present in prepend - skipping profiling")
            MIN_META = False

    pipeline_type = pipeline_type.lower()
    pipeline: Pipeline = None
    pipelines: List[Pipeline] = []
    if pipeline_type == "table":
        if MIN_META:
            pipelines.append(
                tabular.create_pipeline(
                    metric=metric,
                    resolver=resolver,
                    **pipeline_info,
                    profiler="simple",
                    use_boost=True,
                    grid_search=not tune_pipeline
                )
            )
            pipelines.append(
                tabular.create_pipeline(
                    metric=metric,
                    resolver=resolver,
                    **pipeline_info,
                    profiler="simple",
                    use_boost=False,
                    grid_search=not tune_pipeline
                )
            )
            pipelines.append(
                tabular.create_pipeline(
                    metric=metric,
                    resolver=resolver,
                    **pipeline_info,
                    profiler="simon",
                    use_boost=False,
                    grid_search=not tune_pipeline
                )
            )

            pipelines.append(
                tabular.create_pipeline(
                    metric=metric,
                    resolver=resolver,
                    **pipeline_info,
                    profiler="simon",
                    use_boost=True,
                    grid_search=not tune_pipeline
                )
            )
        else:
            pipelines.append(
                tabular.create_pipeline(
                    metric=metric,
                    resolver=resolver,
                    **pipeline_info,
                    profiler="none",
                    use_boost=True,
                )
            )
            pipelines.append(
                tabular.create_pipeline(
                    metric=metric,
                    resolver=resolver,
                    **pipeline_info,
                    profiler="none",
                    use_boost=False,
                    grid_search=not tune_pipeline
                )
            )
    elif pipeline_type == "graph_matching":
        pipelines.append(
            graph_matching.create_pipeline(metric=metric, resolver=resolver)
        )
    elif pipeline_type == "timeseries_classification":
        pipelines.append(
            timeseries_kanine.create_pipeline(metric=metric, resolver=resolver)
        )
        if gpu:
            pipelines.append(
                timeseries_lstm_fcn.create_pipeline(
                    metric=metric, resolver=resolver, **pipeline_info
                )
            )
    elif pipeline_type == "question_answering":
        if gpu:
            pipelines.append(
                question_answer.create_pipeline(metric=metric, resolver=resolver)
            )
        pipelines.append(tabular.create_pipeline(metric=metric, resolver=resolver))
    elif pipeline_type == "text":
        pipelines.append(
            text.create_pipeline(metric=metric, resolver=resolver, **pipeline_info)
        )
        pipelines.append(
            text_sent2vec.create_pipeline(
                metric=metric, resolver=resolver, **pipeline_info
            )
        )
    elif pipeline_type == "image":
        pipelines.append(
            image.create_pipeline(
                metric=metric, resolver=resolver, **pipeline_info, sample=True
            )
        )
        pipelines.append(
            image.create_pipeline(metric=metric, resolver=resolver, **pipeline_info)
        )

    elif pipeline_type == "remote_sensing":
        pipelines.append(
            remote_sensing.create_pipeline(
                metric=metric, resolver=resolver, grid_search=True, **pipeline_info
            )
        )
        pipelines.append(
            image.create_pipeline(metric=metric, resolver=resolver, **pipeline_info)
        )

    elif pipeline_type == "object_detection":
        # pipelines.append(
        #     object_detection.create_pipeline(
        #         metric=metric, resolver=resolver
        #     ))
        pipelines.append(
            object_detection_yolo.create_pipeline(metric=metric, resolver=resolver)
        )
    elif pipeline_type == "audio":
        pipelines.append(audio.create_pipeline(metric=metric, resolver=resolver))
    elif pipeline_type == "collaborative_filtering":
        if gpu:
            pipelines.append(
                collaborative_filtering.create_pipeline(
                    metric=metric, resolver=resolver, **pipeline_info
                )
            )
        pipelines.append(tabular.create_pipeline(metric=metric, resolver=resolver))
    elif pipeline_type == "vertex_nomination":
        pipelines.append(
            vertex_nomination.create_pipeline(metric, resolver, **pipeline_info)
        )
        # pipelines.append(
        #     vertex_nomination_jhu.create_pipeline(metric, resolver, **pipeline_info)
        # )
    elif pipeline_type == "vertex_classification":
        # force using vertex classification
        # TODO - should determine the graph data format
        pipelines.append(
            vertex_classification.create_pipeline(metric=metric, resolver=resolver)
        )
    elif pipeline_type == "link_prediction":
        pipelines.append(
            link_prediction.create_pipeline(metric=metric, resolver=resolver)
        )
        pipelines.append(
            link_prediction_jhu.create_pipeline(metric=metric, resolver=resolver)
        )
    elif pipeline_type == "community_detection":
        pipelines.append(
            community_detection.create_pipeline(metric=metric, resolver=resolver)
        )
    elif pipeline_type == "clustering":
        n_clusters = problem["inputs"][0]["targets"][0]["clusters_number"]
        col_name = problem["inputs"][0]["targets"][0]["column_name"]
        pipelines.append(
            clustering.create_pipeline(
                metric,
                num_clusters=n_clusters,
                cluster_col_name=col_name,
                resolver=resolver,
            )
        )
    elif pipeline_type == "timeseries_forecasting":
        pipelines.append(
            timeseries_var.create_pipeline(metric=metric, resolver=resolver)
        )
        if gpu:
            pipelines.append(
                timeseries_deepar.create_pipeline(metric=metric, resolver=resolver)
            )
            # avoid CUDA OOM by not using it.
            # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    elif pipeline_type == "semisupervised_tabular":
        exclude_column = problem["inputs"][0]["targets"][0]["column_index"]
        pipelines.append(
            semisupervised_tabular.create_pipeline(
                metric=metric,
                resolver=resolver,
                exclude_column=exclude_column,
                profiler="simon",
            )
        )
        pipelines.append(
            semisupervised_tabular.create_pipeline(
                metric=metric,
                resolver=resolver,
                exclude_column=exclude_column,
                profiler="simple",
            )
        )
    elif pipeline_type == "data_augmentation_tabular":
        pipelines.append(
            data_augmentation_tabular.create_pipeline(
                metric, dataset=train_dataset, keywords=pipeline_info, resolver=resolver
            )
        )
    else:
        logger.error(f"Pipeline type [{pipeline_type}] is not yet supported.")
        return None, train_dataset, []

    # prepend to the base pipeline
    if prepend is not None:
        pipelines_prepend: List[Tuple[pipeline.Pipeline, List[int]]] = []
        for pipeline in pipelines:
            pipeline = _prepend_pipeline(pipeline, prepend)
            pipelines_prepend.append(pipeline)
        pipelines = pipelines_prepend

    tuned_pipelines = []
    scores = []
    for i, pipeline in enumerate(pipelines):
        # tune the pipeline if tuning info was generated, and tuning is enabled
        if len(pipeline[1]) > 0  and tune_pipeline:
            try:
                pipeline = hyperparam_tune(
                    pipeline,
                    problem,
                    train_dataset,
                    timeout=min((time_limit / (len(pipelines) + 1)), 600),
                )
                if pipeline is not None:
                    tuned_pipelines.append(pipeline[0])
                    if pipeline[1] is None:
                        scores.append(np.nan)
                    else:
                        scores.append(pipeline[1])
                else:
                    # if timeout return base pipeline
                    tuned_pipelines.append(pipelines[i][0])
                    scores.append(np.nan)
            except Exception as e:
                # if anything happens just return base pipeline
                tuned_pipelines.append(pipeline[0])
                scores.append(np.nan)

        else:
            tuned_pipelines.append(pipeline[0])
            scores.append(np.nan)

    # tuned_pipelines = [pipeline[0] for pipeline in pipelines]
    # scores = np.arange(len(tuned_pipelines))

    ranks: List[float] = []
    for i in np.argsort(scores):
        ranks.append(int(i + 1))

    return tuned_pipelines, train_dataset, ranks


def fit(
    pipeline: pipeline.Pipeline,
    problem: problem.Problem,
    input_dataset: container.Dataset,
    is_standard_pipeline=True,
) -> Tuple[Optional[runtime.Runtime], Optional[runtime.Result]]:
    hyperparams = None
    random_seed = 0
    volumes_dir = config.D3MSTATICDIR

    fitted_runtime, _, result = runtime.fit(
        pipeline,
        [input_dataset],
        problem_description=problem,
        hyperparams=hyperparams,
        random_seed=random_seed,
        volumes_dir=volumes_dir,
        context=metadata_base.Context.TESTING,
        runtime_environment=pipeline_run.RuntimeEnvironment(),
        is_standard_pipeline=is_standard_pipeline,
    )

    if result.has_error():
        raise result.error

    return fitted_runtime, result


def produce(
    fitted_pipeline: runtime.Runtime, input_dataset: container.Dataset
) -> runtime.Result:
    _, result = runtime.produce(
        fitted_pipeline, [input_dataset], expose_produced_outputs=True
    )
    if result.has_error():
        raise result.error
    return result


def produce_pipeline(
    fitted_pipeline: runtime.Runtime, input_dataset: container.Dataset
) -> runtime.Result:
    output, result = runtime.produce(
        fitted_pipeline, [input_dataset], expose_produced_outputs=True
    )
    if result.has_error():
        raise result.error
    return output, result


def is_fully_specified(prepend: pipeline.Pipeline) -> bool:
    # if there's a pipeline and it doesn't have a placeholder then its fully specified
    return prepend and not [
        True for s in prepend.steps if isinstance(s, PlaceholderStep)
    ]


def _prepend_pipeline(
    base: Tuple[pipeline.Pipeline, List[int]], prepend: pipeline.Pipeline
) -> Tuple[pipeline.Pipeline, List[int]]:

    # make a copy of the prepend
    prepend = copy.deepcopy(prepend)

    # find the placeholder node
    replace_index = -1
    for i, prepend_step in enumerate(prepend.steps):
        if isinstance(prepend_step, pipeline.PlaceholderStep):
            replace_index = i
            break

    if replace_index < 0:
        logger.warn(
            f"Failed to prepend pipeline {prepend.id} - continuing with base unmodified"
        )
        return base

    # update prepend outputs to use those from base pipeline
    updated_outputs: List[Dict[str, str]] = []
    for base_output in base[0].outputs:
        # use the base pipeline output but increase the index to account for the appended base pipeline length
        base_ref = base_output["data"].split(".")
        updated_base_ref = (
            f"{base_ref[0]}.{int(base_ref[1]) + replace_index}.{base_ref[2]}"
        )
        updated_outputs.append({"data": updated_base_ref, "name": base_output["name"]})
    prepend.outputs = updated_outputs

    # add each of the base steps, updating their references as necessary
    for base_idx, base_step in enumerate(base[0].steps):
        base_step.index = replace_index + base_idx
        if base_idx == 0:
            prepend.steps[base_step.index] = base_step
        else:
            prepend.steps.append(base_step)

        # update the inputs to account account for concatenation
        for arg_name, arg_val in base_step.arguments.items():
            arg_ref = arg_val["data"].split(".")
            if arg_ref[0] == "inputs":
                # if this was taking inputs, take the last prepend node instead
                arg_val["data"] = f"steps.{int(arg_ref[1]) + replace_index - 1}.produce"
            else:
                arg_val[
                    "data"
                ] = f"{arg_ref[0]}.{int(arg_ref[1]) + replace_index}.{arg_ref[2]}"

    # offset the tuning references to account for the prepend
    tune_steps = [tune_idx + replace_index for tune_idx in base[1]]
    return (prepend, tune_steps)


def _use_gpu() -> bool:
    # check for gpu presence - exception can be thrown when none available depending on
    # system config
    use_gpu = False
    try:
        gpus = GPUtil.getGPUs()
        logger.info(f"{len(gpus)} GPUs detected.  Requested GPU use is [{config.GPU}].")
        if (config.GPU == "auto" or config.GPU == "true") and len(gpus) > 0:
            use_gpu = True
        else:
            use_gpu = False
    except Exception as error:
        use_gpu = False
    logger.info(f"GPU enabled pipelines {use_gpu}")
    return use_gpu


def get_pipeline_hyperparams(pipeline, tune_steps):
    parameters = []
    defaults = {}
    current_step = 0
    black_list_primtives = [
        ExtractColumnsBySemanticTypesPrimitive,
        DatasetToDataFramePrimitive,
    ]
    for i, step in enumerate(pipeline.steps):
        primitive = step.primitive
        hyperparams_class = primitive.metadata.get_hyperparams()
        if primitive not in black_list_primtives and i in tune_steps:
            for name in list(hyperparams_class.configuration.keys()):
                # don't touch fixed hyperparams #TODO is this right?
                if name in pipeline.steps[current_step].hyperparams:
                    default_hyperparam = pipeline.steps[current_step].hyperparams[name]
                else:
                    default_hyperparam = hyperparams_class.configuration[name]._default
                if type(default_hyperparam) == dict:
                    default_hyperparam = default_hyperparam["data"]
                if type(default_hyperparam).__module__ == np.__name__:
                    default_hyperparam = (
                        default_hyperparam.item()
                    )  # can't encode numpy types in mongo

                if name not in list(pipeline.steps[0].hyperparams.keys()):
                    # base on structural type set pyshac types
                    # if in Enumeration, Uniform, UniformInt, UniformBool, Union Set?
                    if (
                        type(hyperparams_class.configuration[name])
                        == hyperparams.UniformInt
                    ):
                        # TODO fix max call
                        upper = hyperparams_class.configuration[name].upper
                        if upper == None:
                            upper = 100  # we don't have a good way to know a reasonable range if not given
                        lower = hyperparams_class.configuration[name].lower
                        if lower == None:
                            lower = 0  # we don't have a good way to know a reasonable range if not given
                        parameters.append(
                            sherpa.Discrete(
                                f"step___{current_step}___{name}",
                                [lower, min(10000 + lower, upper)],
                            )
                        )
                        defaults.update(
                            {f"step___{current_step}___{name}": default_hyperparam}
                        )
                    elif (
                        type(hyperparams_class.configuration[name])
                        == hyperparams.UniformBool
                    ):
                        parameters.append(
                            sherpa.Choice(
                                f"step___{current_step}___{name}", [True, False]
                            )
                        )
                        defaults.update(
                            {f"step___{current_step}___{name}": default_hyperparam}
                        )
                    elif (
                        type(hyperparams_class.configuration[name])
                        == hyperparams.Enumeration
                    ):
                        parameters.append(
                            sherpa.Choice(
                                f"step___{current_step}___{name}",
                                list(hyperparams_class.configuration[name].values),
                            )
                        )
                        defaults.update(
                            {f"step___{current_step}___{name}": default_hyperparam}
                        )
                    elif (
                        type(hyperparams_class.configuration[name])
                        == hyperparams.Uniform
                    ):
                        upper = hyperparams_class.configuration[name].upper
                        if upper == None:
                            upper = 100  # we don't have a good way to know a reasonable range if not given
                        lower = hyperparams_class.configuration[name].lower
                        if lower == None:
                            lower = 0  # we don't have a good way to know a reasonable range if not given
                        parameters.append(
                            sherpa.Continuous(
                                f"step___{current_step}___{name}", [lower, upper]
                            )
                        )
                        defaults.update(
                            {f"step___{current_step}___{name}": default_hyperparam}
                        )
                    elif (
                        type(hyperparams_class.configuration[name])
                        == hyperparams.Bounded
                    ):
                        upper = hyperparams_class.configuration[name].upper
                        if upper == None:
                            upper = 100  # we don't have a good way to know a reasonable range if not given
                        lower = hyperparams_class.configuration[name].lower
                        if lower == None:
                            lower = 0  # we don't have a good way to know a reasonable range if not given
                        if (
                            hyperparams_class.configuration[name].structural_type
                            == float
                        ):
                            parameters.append(
                                sherpa.Continuous(
                                    f"step___{current_step}___{name}", [lower, upper]
                                )
                            )  # todo need an upper limit on bounded
                            defaults.update(
                                {f"step___{current_step}___{name}": default_hyperparam}
                            )
                        else:
                            parameters.append(
                                sherpa.Discrete(
                                    f"step___{current_step}___{name}", [lower, 10]
                                )
                            )  # todo need an upper limit on bounded
                            defaults.update(
                                {f"step___{current_step}___{name}": default_hyperparam}
                            )
                    elif type(hyperparams_class.configuration[name]) == hyperparams.Set:
                        pass  # sets tend to be parse configurations
                    elif (
                        type(hyperparams_class.configuration[name]) == hyperparams.Union
                    ):
                        pass  # union is multiple hyperparams types?
        else:
            for name in list(hyperparams_class.configuration.keys()):
                if name in pipeline.steps[current_step].hyperparams:
                    default_hyperparam = pipeline.steps[current_step].hyperparams[name]
                else:
                    default_hyperparam = hyperparams_class.configuration[name]._default
                if type(default_hyperparam).__module__ == np.__name__:
                    default_hyperparam = (
                        default_hyperparam.item()
                    )  # can't encode numpy types in mongo
                if type(default_hyperparam) == dict:
                    default_hyperparam = default_hyperparam["data"]
                parameters.append(
                    sherpa.Choice(
                        f"step___{current_step}___{name}", [default_hyperparam],
                    )
                )
                defaults.update({f"step___{current_step}___{name}": default_hyperparam})

        current_step += 1
    return parameters, [defaults]


def hyperparam_tune(pipeline, problem, dataset, timeout=600):
    # train test split dataset
    tune_steps = pipeline[1]
    pipeline = pipeline[0]
    with open("current_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    with open("problem.pkl", "wb") as f:
        pickle.dump(problem, f)

    metric_map: typing.Dict[PerformanceMetricBase, bool] = {
        PerformanceMetric.ACCURACY: False,
        PerformanceMetric.PRECISION: False,
        PerformanceMetric.RECALL: False,
        PerformanceMetric.F1: False,
        PerformanceMetric.F1_MICRO: False,
        PerformanceMetric.F1_MACRO: False,
        PerformanceMetric.MEAN_SQUARED_ERROR: True,
        PerformanceMetric.ROOT_MEAN_SQUARED_ERROR: True,
        PerformanceMetric.MEAN_ABSOLUTE_ERROR: True,
        PerformanceMetric.R_SQUARED: True,
        PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION: False,
        PerformanceMetric.JACCARD_SIMILARITY_SCORE: False,
        PerformanceMetric.PRECISION_AT_TOP_K: False,
        PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION: False,
        PerformanceMetric.HAMMING_LOSS: True,
        PerformanceMetric.MEAN_RECIPROCAL_RANK: False,
        PerformanceMetric.HITS_AT_K: False,
    }
    performance_metric_ref = problem["problem"]["performance_metrics"][0]
    lower_is_better = metric_map[performance_metric_ref["metric"]]
    params, defaults = get_pipeline_hyperparams(pipeline, tune_steps)
    #alg = sherpa.algorithms.GPyOpt(initial_data_points=defaults, max_num_trials=128)
    alg = sherpa.algorithms.LocalSearch(seed_configuration=defaults[0])
    alg.next_trial = None
    scheduler = sherpa.schedulers.LocalScheduler()
    stopping_rule = sherpa.algorithms.MedianStoppingRule(min_iterations=1, min_trials=5)

    def run_sherpa_optimize(fun, **kwargs):
        fun(**kwargs)

    p = Process(
        target=run_sherpa_optimize,
        args=(sherpa.optimize,),
        kwargs={
            "parameters": params,
            "algorithm": alg,
            "stopping_rule": stopping_rule,  # todo this isn't working
            "lower_is_better": lower_is_better,
            "command": "python3 ./processing/run_sherpa.py",
            # "filename": "./processing/run_sherpa.py",
            "output_dir": f"./sherpa_temp/{pipeline.id}",
            "scheduler": scheduler,
            "max_concurrent": 8,
            "verbose": 2,
            "db_port": 27017,
        },
        name="hyperparameter tune",
    )

    p.start()
    start = time.time()
    while time.time() - start <= timeout:

        if not p.is_alive():
            logger.info("All the processes are done, break now.")
            p.join()
            break

        time.sleep(1)  # Just to avoid hogging the CPU

    else:
        # We only enter this if we didn't 'break' above.
        logger.info("timed out, killing all processes")
        logger.info(scheduler.jobs)
        # p.join(1)
        # if p.is_alive():
        p.terminate()
        p.join()

        logger.info("timeout on {final_result}")
        # make sure mongo is shut down

        client = pymongo.MongoClient(port=27017)
        db = client.sherpa
        # try:
        #     logger.info("trying to close mongo from db eval")
        #     db.eval("db.getSiblingDB('admin').shutdownServer({ 'force' : true })")
        #     time.sleep(2)
        # except pymongo.errors.ServerSelectionTimeoutError:
        # try closing mongo using os
        logger.info("closing mongo from os")
        for p in psutil.process_iter(attrs=["pid", "name"]):
            if "mongod" in p.info["name"]:
                print(p.info)
                os.kill(p.info["pid"], signal.SIGKILL)
                time.sleep(5)

    all_subdirs = [
        os.path.join("./sherpa_temp", d) for d in os.listdir("./sherpa_temp")
    ]
    output_dir = max(all_subdirs, key=os.path.getmtime)

    if os.path.isfile(
        os.path.join(output_dir, "results.csv")
    ):  # there were result before timeout
        results = pd.read_csv(os.path.join(output_dir, "results.csv"))
        final_result = alg.get_best_result(
            parameters=None, results=results, lower_is_better=lower_is_better
        )
        # clean up

    else:
        final_result = {"Objective": None}
        final_result.update(defaults[0])

    if final_result.get("Objective", 9e5) == 9e5:
        return None
    # recreate final pipeline
    final_pipeline = pipeline
    step_params = defaultdict(dict)
    for name, param in final_result.items():
        if name.startswith("step"):
            step = name.split("___")[1]
            step_params[step].update({name.split("___")[2]: param})
        for i, step in enumerate(final_pipeline.steps):
            if step_params[str(i)] != {}:
                step.hyperparams = {}
                if i > 0 and i < len(final_pipeline.steps):
                    try:
                        for name, value in step_params[str(i)].items():
                            if type(value).__module__ == np.__name__:
                                value = (
                                    value.item()
                                )  # pandas stores things in numpy types.
                            if value in ["()", "[]"]:
                                continue
                            elif value == None:
                                continue
                            elif value == "nan":
                                value = None
                            elif type(value) == float:
                                if math.isnan(value):
                                    continue
                            elif type(value) == str:
                                if value.startswith("(") and value.endswith(")"):
                                    # tuple as string. todo figure out another way to parse strings
                                    value = tuple(
                                        value.replace("'", "")[1:-1].split(",")
                                    )
                                    value = tuple(x.strip() for x in value)
                                elif value.startswith("[") and value.endswith("]"):
                                    # tuple as string. todo figure out another way to parse strings
                                    value = tuple(
                                        value.replace("'", "")[1:-1].split(",")
                                    )
                                    # todo check what type it expects
                                    value = list(
                                        int(x.strip())
                                        if x.strip().isdigit()
                                        else x.strip()
                                        for x in value
                                    )
                            step.add_hyperparameter(
                                name=name, argument_type=ArgumentType.VALUE, data=value
                            )
                    except Exception as e:
                        # import pdb
                        # pdb.set_trace()
                        logger.error(f"failed to recreate final pipeline - {e.msg}")
    if final_result.get("Objective") is None:
        return None
    return (
        final_pipeline,
        final_result.get("Objective", 0) * {True: 1, False: -1}[lower_is_better],
    )
