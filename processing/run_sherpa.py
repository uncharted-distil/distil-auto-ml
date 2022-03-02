#
#   Copyright © 2021 Uncharted Software Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import pickle
import sys
import traceback
import typing
from collections import defaultdict
from typing import Tuple, Optional

from sherpa import Client
from d3m import container, runtime, primitives
from d3m import index
from d3m.metadata import base as metadata_base, pipeline, problem, pipeline_run
from d3m.metadata.base import ArgumentType
from d3m.metadata.problem import PerformanceMetricBase, PerformanceMetric
from d3m.metrics import class_map
from sklearn.model_selection import train_test_split
import numpy as np
import logging

logger = logging.getLogger(__name__)


def fit(
    pipeline: pipeline.Pipeline,
    problem: problem.Problem,
    input_dataset: container.Dataset,
    is_standard_pipeline=True,
) -> Tuple[Optional[runtime.Runtime], Optional[runtime.Result]]:
    hyperparams = None
    random_seed = 0
    volumes_dir = os.getenv("D3MSTATICDIR", "/static")

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


def produce_pipeline(
    fitted_pipeline: runtime.Runtime, input_dataset: container.Dataset
) -> runtime.Result:
    output, result = runtime.produce(
        fitted_pipeline, [input_dataset], expose_produced_outputs=True
    )
    if result.has_error():
        raise result.error
    return output, result


def main(client, trial):
    # Create new model.
    try:
        with open("current_pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        with open("dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
        with open("problem.pkl", "rb") as f:
            problem = pickle.load(f)

        # split = int(len(dataset["learningData"]) * 0.8)
        train_dataset = dataset.copy()
        test_dataset = dataset.copy()
        train_dataset["learningData"], test_dataset["learningData"] = train_test_split(
            train_dataset["learningData"], random_state=42, shuffle=False
        )
        train_dataset["learningData"] = train_dataset["learningData"].reset_index(
            drop=True
        )
        test_dataset["learningData"] = test_dataset["learningData"].reset_index(
            drop=True
        )
        # test_dataset = dataset.copy()
        # train_dataset["learningData"] = train_dataset["learningData"][:split]
        # # train_dataset['0'] = train_dataset['0'][:split]
        # test_dataset["learningData"] = test_dataset["learningData"][split:].reset_index(
        #     drop=True
        # )

        metric_map: typing.Dict[PerformanceMetricBase, bool] = {
            PerformanceMetric.ACCURACY: -1,
            PerformanceMetric.PRECISION: -1,
            PerformanceMetric.RECALL: -1,
            PerformanceMetric.F1: -1,
            PerformanceMetric.F1_MICRO: -1,
            PerformanceMetric.F1_MACRO: -1,
            PerformanceMetric.MEAN_SQUARED_ERROR: 1,
            PerformanceMetric.ROOT_MEAN_SQUARED_ERROR: 1,
            PerformanceMetric.MEAN_ABSOLUTE_ERROR: 1,
            PerformanceMetric.R_SQUARED: 1,
            PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION: -1,
            PerformanceMetric.JACCARD_SIMILARITY_SCORE: -1,
            PerformanceMetric.PRECISION_AT_TOP_K: -1,
            PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION: -1,
            PerformanceMetric.HAMMING_LOSS: 1,
            PerformanceMetric.MEAN_RECIPROCAL_RANK: -1,
            PerformanceMetric.HITS_AT_K: -1,
            PerformanceMetric.ROC_AUC: -1,
            PerformanceMetric.ROC_AUC_MACRO: -1,
            PerformanceMetric.ROC_AUC_MICRO: -1,
        }

        performance_metric_ref = problem["problem"]["performance_metrics"][0]
        lower_is_better_sign = metric_map[performance_metric_ref["metric"]]

        trial_pipeline = pipeline
        step_params = {}
        for name, param in trial.parameters.items():
            if name.startswith("step"):
                step = name.split("___")[1]
                if step not in step_params:
                    step_params[step] = {}
                step_params[step].update({name.split("___")[2]: param})
        for i, step in enumerate(trial_pipeline.steps):
            if str(i) in step_params:
                if step.primitive == index.get_primitive(
                    "d3m.primitives.operator.dataset_map.DataFrameCommon"
                ):
                    step.hyperparams = {
                        "primitive": {
                            "type": ArgumentType.PRIMITIVE,
                            "data": step.index - 1,
                        }
                    }
                elif step_params[str(i)] != {}:
                    step.hyperparams = {}
                    if i > 0 and i < len(trial_pipeline.steps):
                        for name, value in step_params[str(i)].items():
                            step.add_hyperparameter(
                                name=name, argument_type=ArgumentType.VALUE, data=value
                            )
            print(step.hyperparams)

        fitted_pipeline, predictions = fit(trial_pipeline, problem, train_dataset)
        performance_metric_ref = problem["problem"]["performance_metrics"][0]
        if "params" in performance_metric_ref:
            performance_metric = class_map[performance_metric_ref["metric"]](
                **performance_metric_ref["params"]
            )
        else:
            performance_metric = class_map[performance_metric_ref["metric"]]()
        predictions, _ = produce_pipeline(fitted_pipeline, test_dataset)
        predictions["d3mIndex"] = predictions["d3mIndex"].astype(int)
        print(predictions)
        true_data = test_dataset["learningData"][
            [x for x in predictions.columns if x != "confidence"]
        ]
        true_data["d3mIndex"] = true_data["d3mIndex"].astype(int)
        true_data = true_data[
            true_data["d3mIndex"].isin(predictions["d3mIndex"])
        ]  # todo why are these different?
        # make sure that true_Data and predictions are of the same type.
        print(true_data)
        if "confidence" in predictions.columns:
            label_col = predictions.columns[-2]
        else:
            label_col = predictions.columns[-1]
        try:
            true_data = true_data.astype(predictions[label_col].dtype)
        except ValueError:
            true_data = true_data.astype(float)
            true_data = true_data.astype(int)
        score = performance_metric.score(true_data, predictions)
    except Exception as e:
        print(f"Error on pipeline trail {trial} : {e}")
        print("-" * 60)
        traceback.print_exc(file=sys.stdout)
        score = 9e5 * lower_is_better_sign

    client.send_metrics(trial=trial, iteration=i + 1, objective=score)


if __name__ == "__main__":
    client = Client(port=27017, host="localhost")
    trial = client.get_trial()
    main(client, trial)
