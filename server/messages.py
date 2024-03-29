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

import pathlib
from api import utils as api_utils
from google.protobuf import json_format

import config
import random
from api import core_pb2, problem_pb2, value_pb2, pipeline_pb2, primitive_pb2, utils

from d3m.metadata import pipeline

from d3m import index

# List of value types the we support for returning data to the TA3
ALLOWED_TYPES = set([api_utils.ValueType.CSV_URI, api_utils.ValueType.PARQUET_URI])


class Messaging:
    def get_dataset_uri(self, msg):
        if len(msg.inputs) == 0:
            return False
        return [i.dataset_uri or i.csv_uri for i in msg.inputs]

    def get_problem_type(self, msg):
        _type = msg.problem.problem.task_keywords
        if len(_type) == 0:
            return False

        # this seems to only be used for validation, not used in router.
        _type_str = _type[0].upper()
        return _type_str

    def get_search_id(self, msg):
        return msg.search_id

    def make_get_search_solutions_result(self, solution_id, progress_msg, rank):

        # if no solutin_id is set this is just a progress message and doesn't need
        # score info
        scores = []

        if solution_id is not None:
            scores = [
                core_pb2.SolutionSearchScore(
                    scoring_configuration=core_pb2.ScoringConfiguration(
                        method="RANKING"
                    ),
                    scores=[
                        core_pb2.Score(
                            metric=problem_pb2.ProblemPerformanceMetric(
                                metric="RANK",
                            ),
                            value=value_pb2.Value(raw=value_pb2.ValueRaw(int64=rank)),
                        )
                    ],
                )
            ]

        m = core_pb2.GetSearchSolutionsResultsResponse(
            solution_id=solution_id,
            progress=progress_msg,
            internal_score=0.0,
            scores=scores,
        )
        return m

    def search_solutions_response(self, search_id):
        m = core_pb2.SearchSolutionsResponse(search_id=search_id)
        return m

    def score_solution_response(self, request_id):
        return core_pb2.ScoreSolutionResponse(request_id=request_id)

    def unpack_score_solution_request(self, request):
        # empty string returned if solution_id doesn't exist
        solution_id = request.solution_id

        # metric (need at least one)
        metrics = [
            m.metric.lower()
            for m in request.performance_metrics
            if len(m.metric) is not 0
        ]

        pos_labels = [
            m.pos_label
            for m in request.performance_metrics
            if len(m.pos_label) is not 0
        ]


        # method required to be defined
        method = request.configuration.method.lower()
        if len(method) == 0:
            method = False

        # Optionals
        # uri is optional because it will score on training data if there isn't one in request
        try:
            dataset_uri = self.get_dataset_uri(request)
        except:
            dataset_uri = None

        folds = request.configuration.folds
        train_test_ratio = request.configuration.train_test_ratio
        shuffle = request.configuration.shuffle
        random_seed = request.configuration.random_seed
        stratified = request.configuration.stratified

        r = (
            solution_id,
            dataset_uri,
            metrics,
            pos_labels,
            method,
            folds,
            train_test_ratio,
            shuffle,
            random_seed,
            stratified,
        )

        return r

    def make_get_score_solution_results_response(self, scores, progess_msg):
        m = core_pb2.GetScoreSolutionResultsResponse(
            scores=scores, progress=progess_msg
        )
        return m

    def make_score_message(self, metric, value):
        inner_metric = metric.upper()
        metric = problem_pb2.ProblemPerformanceMetric(metric=inner_metric)
        # TODO: obviously way more
        raw = value_pb2.ValueRaw(double=value)
        val = value_pb2.Value(raw=raw)
        score = core_pb2.Score(metric=metric, value=val)
        return score

    def make_hello_response_message(self):
        resp = core_pb2.HelloResponse(
            user_agent=config.SERVER_USER_AGENT,
            version=core_pb2.DESCRIPTOR.GetOptions().Extensions[
                core_pb2.protocol_version
            ],
        )
        resp.allowed_value_types.append("PARQUET_URI")
        resp.allowed_value_types.append("CSV_URI")
        resp.allowed_value_types.append("DATASET_URI")
        resp.allowed_value_types.append("RAW")
        return resp

    def get_solution_id(self, message):
        return message.solution_id

    def get_fit_solution_id(self, message):
        return message.solution_id

    def get_fitted_solution_id(self, message):
        return message.fitted_solution_id

    def get_solution_uri(self, message):
        return message.solution_uri

    def get_fitted_solution_uri(self, message):
        return message.fitted_solution_uri

    def make_fit_solution_message(self, request_id):
        return core_pb2.FitSolutionResponse(request_id=request_id)

    def make_get_fit_solution_results_response(
        self, fitted_solution_id, progess_msg, output_key_map=None
    ):
        if output_key_map:
            fb_output_map = {
                output_id: value_pb2.Value(csv_uri=csv_uri)
                for (output_id, csv_uri) in output_key_map.items()
            }
            resp = core_pb2.GetFitSolutionResultsResponse(
                fitted_solution_id=fitted_solution_id,
                progress=progess_msg,
                exposed_outputs=fb_output_map,
            )
        else:
            resp = core_pb2.GetFitSolutionResultsResponse(
                fitted_solution_id=fitted_solution_id, progress=progess_msg
            )
        return resp

    def get_output_keys(self, message):
        output_keys = list(message.expose_outputs)
        if len(output_keys) != 0:
            return output_keys
        else:
            return None

    def get_output_types(self, message):
        output_types = list(message.expose_value_types)
        if len(output_types) != 0:
            return output_types
        else:
            return None

    def unpack_produce_solution_request(self, message):
        fitted_solution_id = self.get_fitted_solution_id(message)
        dataset_uri = self.get_dataset_uri(message)
        expose_output_keys = self.get_output_keys(message)
        output_types = self.get_output_types(message)
        return fitted_solution_id, dataset_uri, expose_output_keys, output_types

    def unpack_fit_solution_request(self, message):
        solution_id = self.get_solution_id(message)
        dataset_uri = self.get_dataset_uri(message)
        expose_output_keys = self.get_output_keys(message)
        output_types = self.get_output_types(message)
        return solution_id, dataset_uri, expose_output_keys, output_types

    def make_produce_solution_response(self, request_id):
        return core_pb2.ProduceSolutionResponse(request_id=request_id)

    def make_describe_solution_response(self, description):

        pipeline_obj = pipeline.Pipeline.from_json(description)

        allowed = [
            utils.ValueType.RAW,
            utils.ValueType.CSV_URI,
            utils.ValueType.DATASET_URI,
            utils.ValueType.PICKLE_BLOB,
            utils.ValueType.PICKLE_URI,
        ]
        pipeline_description = utils.encode_pipeline_description(
            pipeline_obj, allowed, "."
        )

        msg = core_pb2.DescribeSolutionResponse(pipeline=pipeline_description)
        return msg

    def make_get_produce_solution_results_response(self, output_key_map, progress_msg):
        pb_output_map = {
            output_id: value_pb2.Value(csv_uri=csv_uri)
            for (output_id, csv_uri) in output_key_map.items()
        }
        rsp = core_pb2.GetProduceSolutionResultsResponse(
            exposed_outputs=pb_output_map, progress=progress_msg
        )
        return rsp

    def get_request_id(self, message):
        return message.request_id

    def get_search_template(self, message):
        return message.template

    def dump_solution_template(self, template):
        return json_format.MessageToDict(template)

    def make_progress_msg(
        self, progress_state, status_msg="", task_start=None, task_end=None
    ):
        """Return a properly built Progress message.

        Args:
            progress_state (str): ProgressState enum value as string, possible values are
                "PROGRESS_UNKNOWN"
                "PENDING"
                "RUNNING"
                "COMPLETED"
                "ERRORED"
            status_msg (str): text description of progress, only use for ERRORED status
            task_start (timestamp): UNUSED
            task_end (timestamp): UNUSED
        """
        progress_msg = core_pb2.Progress(
            state=core_pb2.ProgressState.Value(progress_state),
            status=status_msg,
            start=task_start,
            end=task_start,
        )
        return progress_msg

    def make_solution_export_response(self):
        # SolutionExportResponse is always empty
        return core_pb2.SolutionExportResponse()

    def make_save_fitted_solution_response(self, fitted_solution_uri):
        response_msg = core_pb2.SaveFittedSolutionResponse(
            fitted_solution_uri=fitted_solution_uri
        )
        return response_msg

    def make_save_solution_response(self, solution_uri):
        response_msg = core_pb2.SaveSolutionResponse(solution_uri=solution_uri)
        return response_msg

    def make_load_fitted_solution_response(self, fitted_solution_id):
        response_msg = core_pb2.LoadFittedSolutionResponse(
            fitted_solution_id=fitted_solution_id
        )
        return response_msg

    def make_load_solution_response(self, solution_id):
        response_msg = core_pb2.LoadSolutionResponse(solution_id=solution_id)
        return response_msg

    def get_rank(self, msg):
        return msg.rank

    def make_list_primitives_response(self):
        resp = core_pb2.ListPrimitivesResponse()
        primitives = [
            primitive_pb2.Primitive(
                id=prim_data.metadata.query()["id"],
                version=prim_data.metadata.query()["version"],
                python_path=prim_data.metadata.query()["name"],
                digest=prim_data.metadata.query()["digest"],
            )
            for prim_data in index.get_loaded_primitives()
        ]
        resp.primitives.extend(primitives)
        return resp
