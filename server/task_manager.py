"""
Hello - TaskManager translates between
protobuf messages and DAG tasks for the worker.
"""
import json
import logging
import time
import pathlib
from sqlalchemy import exists, and_

import config

import models
import utils
from server import export
from server.messages import Messaging
from server.validation import RequestValidator

from google.protobuf import json_format

from api import utils as api_utils
from d3m.metadata import pipeline

class TaskManager():
    def __init__(self, servicer):
        self.session = models.get_session(config.DB_LOCATION)
        self.logger = logging.getLogger('distil.TaskManager')
        self.logger.info('Initialized TaskManager')
        self.msg = Messaging()
        self.validator = RequestValidator()
        self.servicer = servicer

    def close(self):
        self.session.close()

    def _generate_id(self):
        _id = utils.generate_id()
        return _id

    def SearchSolutions(self, message):

        """Get potential pipelines and load into DB."""
        # Validate request is in required format, extract if it is
        dataset_uri, problem_proto, template_proto = self.validator.validate_search_solutions_request(message)

        # Generate search ID
        search_id = self._generate_id()

        # serialize the problem buf to json to save it in the db
        if problem_proto:
            problem_obj = api_utils.decode_problem_description(problem_proto)
            problem_json = json.dumps(problem_obj.to_json_structure())
        else:
            problem_json = None

        # serialize the pipeline to a string for storage in db if one is provided
        search_template_json = None
        if template_proto is not None:
            search_template_obj = api_utils.decode_pipeline_description(message.template, pipeline.Resolver(load_all_primitives=False))
            search_template_json = search_template_obj.to_json()

        # Create search row in DB
        search = models.Searches(id=search_id, problem=problem_json, dataset_uri=dataset_uri, search_template=search_template_json)

        self.session.add(search)
        self.session.commit()
        return search_id

    def GetSearchSolutionsResults(self, request):
        """
        Searches for pipelines
        associated with given search id
        """
        search_id = self.validator.validate_get_search_solutions_results_request(request, self.session)
        seen_ids = []

        start = time.time()


        while True:
            pipelines = self.session.query(models.Pipelines) \
                            .filter(models.Pipelines.search_id==str(search_id)) \
                            .filter(models.Pipelines.error == False) \
                            .filter(models.Pipelines.ended == True) \
                            .all()
            if len(pipelines) > 0:
                for pipeline in pipelines:
                # Check if Solution already exists
                    solution = self.session.query(models.Solutions) \
                                           .filter(models.Solutions.search_id==search_id) \
                                           .filter(models.Solutions.pipeline_id==pipeline.id) \
                                           .first()
                    # Generate ValidSolution row if has not
                    # been previously verified
                    if not solution:
                        # Link the task to the solution
                        solution_id = pipeline.id

                        solution = models.Solutions(
                            id=solution_id,
                            search_id=search_id,
                            pipeline_id=pipeline.id)
                        self.session.add(solution)

                    # End session
                    self.session.commit()
                    self.session.refresh(pipeline)
                    progress_msg = self.msg.make_progress_msg("COMPLETED")
                    yield self.msg.make_get_search_solutions_result(pipeline.id, progress_msg, pipeline.rank)
                break
            else:
                yield False




    def ScoreSolution(self, request):
        """
        Create a 'task' for each bit of the score objective
        """
        # validate and unpack message if valid
        # TODO: HAVE SENSIBLE DEFAULTS HERE
        # SO THEY CAN BE STORED IN DB
        (solution_id, dataset_uri, metrics,
        method, folds, train_test_ratio,
        shuffle, random_seed,
        stratified) = self.validator.validate_score_solution_request(request)

        # TODO: if it has been scored already in the same manner,
        # return the request_id from that first time
        score_request_id = self._generate_id()
        solution_id = request.solution_id
        # Create mapping between score_request and solution_id
        _request = models.Requests(
            id=score_request_id,
            solution_id=solution_id,
            type="SCORE")
        self.session.add(_request)
        self.session.commit()

          # Fetch the pipeline record
        _, pipeline_record = self.session.query(models.Solutions, models.Pipelines) \
                                         .filter(models.Solutions.id==solution_id) \
                                         .filter(models.Solutions.pipeline_id==models.Pipelines.id) \
                                         .first()

        # Fetch the search record and extract the problem
        search = self.session.query(models.Searches) \
                             .filter(models.Searches.id==pipeline_record.search_id) \
                             .first()

        # Attempt to create a scoring_config
        # Add a SCORE task per metric in the request
        for metric in metrics:
            conf_id = self._generate_id()
            conf = models.ScoreConfig(
                id=conf_id,
                metric=metric,
                method=method,
                num_folds=folds,
                # TODO: train_test_ratio is currently unused by SolutionScorer
                # remove it or fix it
                train_test_ratio=train_test_ratio,
                shuffle=shuffle,
                random_seed=random_seed,
                stratified=stratified)
            self.session.add(conf)
            task_id = self._generate_id()
            task = models.Tasks(id=task_id,
                                type="SCORE",
                                solution_id=solution_id,
                                dataset_uri=dataset_uri,
                                score_config_id=conf_id,
                                problem=search.problem)
            self.session.add(task)
            # Add configs and tasks to pool
            self.session.commit()

        return score_request_id


    def GetScoreSolutionResults(self, message):
        request_id = self.validator.validate_get_score_solution_results_request(message, self.session)
        seen_ids = []

        solution_id = self.session.query(models.Requests) \
                      .filter(models.Requests.id==request_id) \
                      .first().solution_id

        while True:
            # get all scores associated with the solution_id from the db
            scores = self.session.query(models.Scores) \
                                 .filter(models.Scores.solution_id==solution_id) \
                                 .all()

            # Refresh what we need to
            if scores:
                for s in scores:
                    self.session.refresh(s)

            # TODO: make this a proper list of scores
            if scores:
                score_msgs = []
                for m in scores:
                    config = self.session.query(models.ScoreConfig) \
                                         .filter(models.ScoreConfig.id==m.score_config_id) \
                                         .first()
                    score_msgs.append(self.msg.make_score_message(config.metric, m.value))

                progress_msg = self.msg.make_progress_msg("COMPLETED")

                yield self.msg.make_get_score_solution_results_response(score_msgs, progress_msg)
                break
            else:
                # Force re-query next time
                self.session.expire_all()
                yield False


    def FitSolution(self, message):
        # Generate request ID
        request_id = self._generate_id()

        # Extract the associated solution ID and the dataset URI
        solution_id, dataset_uri = self.validator.validate_fit_solution_request(message)

        # Fetch the pipeline record
        _, pipeline_record = self.session.query(models.Solutions, models.Pipelines) \
                                         .filter(models.Solutions.id==solution_id) \
                                         .filter(models.Solutions.pipeline_id==models.Pipelines.id) \
                                         .first()


        # Fetch the search record and extract the problem
        search = self.session.query(models.Searches) \
                             .filter(models.Searches.id==pipeline_record.search_id) \
                             .first()

        # We need to pass the pipeline json and the fully specified status along as downstream processing
        # needs the extra conext for execution in the runtime.
        pipeline_json = pipeline_record.pipelines
        fully_specified =  pipeline_record.fully_specified

        # add a fit task to the tasks table
        task_id = self._generate_id()
        task = models.Tasks(id=task_id,
                            type="FIT",
                            request_id=request_id,
                            solution_id=solution_id,
                            dataset_uri=dataset_uri,
                            pipeline=pipeline_json,
                            problem=search.problem,
                            fully_specified = fully_specified)
        self.session.add(task)
        self.session.commit()

        # make a record for the request and commit to the database
        request_record = models.Requests(id=request_id,
                                         task_id=task.id,
                                         type="FIT",
                                         solution_id=solution_id)
        self.session.add(request_record)
        self.session.commit()

        return request_id


    def GetFitSolutionResults(self, message):
        request_id = self.validator.validate_get_fit_solution_results_request(message, self.session)

        start = time.time()

        while True:
            task = self.session.query(models.Tasks) \
                                .filter(models.Tasks.request_id==request_id) \
                                .first()
            # refresh emits an immediate SELECT to the database to reload all attributes on task
            # this allows us to get the updates written to the db when the task is completed
            self.session.refresh(task)

            task_complete = task.ended
            if not task_complete:
                self.logger.debug("FIT task not complete, waiting")
                if time.time() - start > config.PROGRESS_INTERVAL:
                    start = time.time()
                    progress_msg = self.msg.make_progress_msg("RUNNING")
                    yield self.msg.make_get_fit_solution_results_response(None, progress_msg)
                else:
                    yield False
            if task_complete:
                # check if the task has an error
                if task.error:
                    raise RuntimeError("FitSolution task didn't complete successfully")

                # make a record of the fit itself
                fit_solution_id = self._generate_id()
                fit_solution = models.FitSolution(id=fit_solution_id,
                                                  solution_id=task.solution_id,
                                                  task_id=task.id)
                self.session.add(fit_solution)
                self.session.commit()

                progress_msg = self.msg.make_progress_msg("COMPLETED")
                yield self.msg.make_get_fit_solution_results_response(fit_solution_id, progress_msg)
                break


    def ProduceSolution(self, message):
        # Generate request ID
        request_id = self._generate_id()

        # Validate request is in required format, extract if it is
        extracted_fields = self.validator.validate_produce_solution_request(message)
        fitted_solution_id, dataset_uri, output_keys = extracted_fields

        # serialize the output key list json for storage
        output_keys_json = json.dumps(output_keys)

        # Get existing fit_solution.id
        fit_solution = self.session.query(models.FitSolution) \
                          .filter(models.FitSolution.id==fitted_solution_id) \
                          .first()

        if fit_solution is None:
            raise ValueError("Fitted solution id {} doesn't exist".format(fitted_solution_id))

        # add a produce task to the tasks table
        task_id = self._generate_id()
        task = models.Tasks(id=task_id,
                            type="PRODUCE",
                            request_id=request_id,
                            fit_solution_id=fitted_solution_id,
                            solution_id=fit_solution.solution_id,
                            dataset_uri=dataset_uri,
                            output_keys=output_keys_json)
        self.session.add(task)
        self.session.commit()

        # make a record for the request and commit to the database
        request_record = models.Requests(id=request_id,
                                         task_id=task.id,
                                         type="PRODUCE",
                                         fit_solution_id=fitted_solution_id)
        self.session.add(request_record)
        self.session.commit()

        return request_id

    def GetProduceSolutionResults(self, message):
        request_id = self.validator.validate_get_produce_solution_results_request(message, self.session)

        while True:
            task = self.session.query(models.Tasks) \
                               .filter(models.Tasks.request_id==request_id) \
                               .first()
            # refresh emits an immediate SELECT to the database to reload all attributes on task
            # this allows us to get the updates written to the db when the task is completed
            self.session.refresh(task)

            task_complete = task.ended
            if not task_complete:
                self.logger.debug("PRODUCE task not complete, waiting")
                yield False
            if task_complete:
                # check if the task has an error
                if task.error:
                    raise RuntimeError("ProduceSolution task didn't complete successfully")

                # build a map of (output_key, URI)
                task_keys = json.loads(task.output_keys)
                output_key_map = {}
                for task_key in task_keys:
                    # generate a uri from the key and make sure the file exists
                    # TODO(jtorrez): predictions filename creation should live somewhere better than utils
                    preds_path = utils.make_preds_filename(task.request_id, output_key=task_key)
                    if not preds_path.exists() and not preds_path.is_file():
                        raise FileNotFoundError("Predictions file {} doesn't exist".format(preds_path))

                    preds_uri = pathlib.Path(preds_path).absolute().as_uri()
                    output_key_map[task_key] = preds_uri

                progress_msg = self.msg.make_progress_msg("COMPLETED")

                yield self.msg.make_get_produce_solution_results_response(output_key_map, progress_msg)
                break


    def DescribeSolution(self, request):
        # Validate the solution_id
        solution_id = self.validator.validate_describe_solution_request(request, self.session)

        _, pipeline = self.session.query(models.Solutions, models.Pipelines) \
                                        .filter(models.Solutions.id==solution_id) \
                                        .filter(models.Solutions.pipeline_id==models.Pipelines.id) \
                                        .first()
        return pipeline.pipelines

    def SolutionExport(self, request):
        """
        Output pipeline JSON for D3M evaluation.
        """
        solution_id, rank = self.validator.validate_solution_export_request(request)

        _, pipeline = self.session.query(models.Solutions, models.Pipelines) \
                                         .filter(models.Solutions.id==solution_id) \
                                         .filter(models.Solutions.pipeline_id==models.Pipelines.id) \
                                         .first()
        export.export(pipeline, rank)
        # export.export_run(pipeline)
        # export.export_predictions(pipeline)

    def SaveSolution(self, request):
        """
        Output pipeline JSON.
        """
        solution_id = self.validator.validate_save_solution_request(request)

        _, pipeline = self.session.query(models.Solutions, models.Pipelines) \
                                         .filter(models.Solutions.id==solution_id) \
                                         .filter(models.Solutions.pipeline_id==models.Pipelines.id) \
                                         .first()
        export.save_pipeline(pipeline)

        return solution_id

    def SaveFittedSolution(self, request):
        """
        Output fitted pipeline.
        """
        fitted_solution_id = self.validator.validate_save_fitted_solution_request(request)

        runtime = self.servicer.get_fitted_runtime(fitted_solution_id)

        export.save_fitted_pipeline(fitted_solution_id, runtime)

        return fitted_solution_id
