import pathlib
from google.protobuf import json_format

import config

from api import core_pb2, problem_pb2, value_pb2, pipeline_pb2, primitive_pb2, utils

from d3m.metadata import pipeline

class Messaging:

    def get_dataset_uri(self, msg):
        if len(msg.inputs) == 0:
            return False
        return msg.inputs[0].dataset_uri

    def get_problem_type(self, msg):
        _type = msg.problem.problem.task_type
        if _type is problem_pb2.TASK_TYPE_UNDEFINED:
            return False
        _type_str = problem_pb2.TaskType.Name(_type).upper()
        return _type_str

    def get_search_id(self, msg):
        return msg.search_id

    def make_get_search_solutions_result(self, solution_id, progess_msg):
        m = core_pb2.GetSearchSolutionsResultsResponse(
            solution_id=solution_id,
            progress=progess_msg,
            internal_score=0.0)
        return m

    def search_solutions_response(self, search_id):
        m = core_pb2.SearchSolutionsResponse(
            search_id=search_id)
        return m

    def score_solution_response(self, request_id):
        return core_pb2.ScoreSolutionResponse(
            request_id=request_id)

    def unpack_score_solution_request(self, request):
        # empty string returned if solution_id doesn't exist
        solution_id = request.solution_id

        # metric (need at least one)
        metrics = [problem_pb2.PerformanceMetric.Name(m.metric).lower() \
                   for m in request.performance_metrics if m.metric is not problem_pb2.METRIC_UNDEFINED]

        # method required to be defined
        method = core_pb2.EvaluationMethod.Name(request.configuration.method).lower()
        if method == 'evaluation_method_undefined':
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
        
        r = (solution_id, dataset_uri, metrics,
        method, folds, train_test_ratio,
        shuffle, random_seed, stratified)
        
        return r

    def make_get_score_solution_results_response(self, scores, progess_msg):
        m = core_pb2.GetScoreSolutionResultsResponse(scores=scores,
                                                     progress=progess_msg)
        return m

    def make_score_message(self, metric, value):
        metric = metric.upper()
        inner_metric = problem_pb2.PerformanceMetric.Value(metric)
        metric = problem_pb2.ProblemPerformanceMetric(
            metric=inner_metric
        )
        # TODO: obviously way more
        raw = value_pb2.ValueRaw(
            double=value)
        val = value_pb2.Value(
            raw=raw)
        score = core_pb2.Score(
            metric=metric,
            value=val) 
        return score

    def make_hello_response_message(self):
        return core_pb2.HelloResponse(
            user_agent=config.SERVER_USER_AGENT)

    def get_solution_id(self, message):
        return message.solution_id
    
    def get_fit_solution_id(self, message):
        return message.solution_id
        
    def get_fitted_solution_id(self, message):
        return message.fitted_solution_id

    def make_fit_solution_message(self, request_id):
        return core_pb2.FitSolutionResponse(
            request_id=request_id)

    def make_get_fit_solution_results_response(self, fitted_solution_id, progess_msg):
        resp = core_pb2.GetFitSolutionResultsResponse(fitted_solution_id=fitted_solution_id,
                                                      progress=progess_msg)
        return resp

    def get_output_key(self, message):
        output_keys = message.expose_outputs
        if len(output_keys) != 0:
            output_key = output_keys[-1]
        else:
            output_key = None
        return output_key

    def unpack_produce_solution_request(self, message):
        fitted_solution_id = self.get_fitted_solution_id(message)
        dataset_uri = self.get_dataset_uri(message)
        expose_output_key = self.get_output_key(message)
        return fitted_solution_id, dataset_uri, expose_output_key

    def make_produce_solution_response(self, request_id):
        return core_pb2.ProduceSolutionResponse(request_id=request_id)

    def make_describe_solution_response(self, description):

        pipeline_obj = pipeline.Pipeline.from_json(description)

        pipeline_description = utils.encode_pipeline_description(pipeline_obj)

        msg = core_pb2.DescribeSolutionResponse(
            pipeline=pipeline_description)
        return msg

    def make_get_produce_solution_results_response(self, preds_fn, output_key, progress_msg):
        # make a proper URI with file:// prefix
        csv_uri = pathlib.Path(preds_fn).absolute().as_uri()
        val = value_pb2.Value(csv_uri=csv_uri)
        rsp = core_pb2.GetProduceSolutionResultsResponse(
                exposed_outputs={output_key: val},
                progress=progress_msg)
        return rsp

    def get_request_id(self, message):
        return message.request_id

    def get_search_template(self, message):
        return message.template

    def dump_solution_template(self, template):
        return json_format.MessageToDict(template)

    def make_progress_msg(self, progress_state, status_msg="", task_start=None, task_end=None):
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
                end=task_start)
        return progress_msg

    def make_solution_export_response(self):
        # SolutionExportResponse is always empty
        return core_pb2.SolutionExportResponse()

    def get_rank(self, msg):
        return msg.rank

