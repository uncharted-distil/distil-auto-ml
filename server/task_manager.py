"""
Hello - TaskManager translates between
protobuf messages and DAG tasks for the worker.
"""
import json
import logging
from sqlalchemy import exists, and_

import config

import models
import utils
from server.export import export
from server.messages import Messaging
from server.validation import RequestValidator

from google.protobuf import json_format

from api import utils as api_utils
from d3m.metadata import pipeline

class TaskManager():
    def __init__(self):
        self.session = models.get_session(config.DB_LOCATION)
        self.logger = logging.getLogger('exline.TaskManager')
        self.logger.info('Initialized TaskManager')
        self.msg = Messaging()
        self.validator = RequestValidator()

    def close(self):
        self.session.close()

    def _generate_id(self):
        _id = utils.generate_id()
        return _id

    def _combine_pipelines(self, placeholder, their_steps, possible_pipelines):
        combined_pipelines = []
        if not placeholder:
            combined_pipelines = [their_steps]
        else:
            placeholder_input = placeholder['placeholder']['inputs'][0]['data']
            pipelines = []
            _, offset, _ = placeholder_input.split('.')
            offset = int(offset)
            for pipe in possible_pipelines:
                combined_steps = their_steps
                for step in pipe['steps']:
                    for call in step['method_calls']:
                        # Check if initial
                        for key, arg in call.get('arguments', {}).items():
                            #call_input = call.get('arguments', {}).get('inputs', '')
                            if arg == 'inputs.0':
                                call['arguments'][key] = placeholder_input
                            if arg.startswith('steps.'):
                                s, i, a = arg.split('.')
                                i = int(i) + offset + 1
                                call['arguments'][key] = '.'.join([s,str(i),a])
                    combined_steps.append(step)
                pipe['steps'] = combined_steps
                combined_pipelines.append(pipe)

        return combined_pipelines

    def _extract_hypers(self, val):
        # Keep going until we get the good stuff
        # _do you... _see_...
        allowed_types = {
            'string': str,
            'int64': int,
            'float': float,
            'bool': bool,
        }
        for k, v in val.items():
            if k in allowed_types.keys():
                return allowed_types[k](v)
            elif k == 'items':
                items = []
                items += [self._extract_hypers(i) for i in v]
                return items
            else:
                return self._extract_hypers(v)
        return {}


    def _translate_template(self, template):
        placeholder = {}
        dumped = self.msg.dump_solution_template(template)

        # Ensure that 'placeholder' is at end, if anywhere
        step_types = [list(s.keys()).pop() for s in dumped['steps']]
        if 'placeholder' in step_types:
            placeholder = dumped['steps'][-1]

        # Put things in order
        adjusted_steps = []
        for step in dumped['steps']:
            prim = step.get('primitive', False)
            if prim:
                hp = {}
                # Assemble hyperparams
                for hype_name, hype_val in prim.get('hyperparams', {}).items():
                    for _, inner_val in hype_val.items():
                        val = self._extract_hypers(inner_val)
                        hp.update({hype_name: {'data': val}})
                base = {
                    "type": "PRIMITIVE",
                    "stage": "PREPROCESS",
                    "primitive" : {
                        "python_path": prim['primitive']['pythonPath'],
                    },
                    "method_calls": [
                        {
                            "name": "produce",
                            "arguments": {
                                "inputs": prim['arguments']['inputs']['container']['data'],
                            }
                        },
                    ],
                    "hyperparams": hp
                }
                adjusted_steps.append(base)

        if 'placeholder' not in step_types:
            dumped['steps'] = adjusted_steps
            adjusted_steps = dumped

        return adjusted_steps, placeholder

    def _get_pipelines(self, message):
        # Assemble possible pipelines
        # and queue as tasks for worker to validate
        problem_type = self.msg.get_problem_type(message)
        possible_pipelines = get_all_pipelines(problem_type)

        try:
            template = self.msg.get_search_template(message)
            adjusted_steps, placeholder = self._translate_template(template)
            possible_pipelines = self._combine_pipelines(placeholder,
                                                         adjusted_steps,
                                                         possible_pipelines)
        except Exception as e:
            self.logger.info(e)
            pass
        return possible_pipelines

    def SearchSolutions(self, message):
        """Get potential pipelines and load into DB."""
        # Validate request is in required format, extract if it is
        dataset_uri, problem_type = self.validator.validate_search_solutions_request(message)
        # Generate search ID
        search_id = self._generate_id()

        prob = json_format.MessageToDict(message.problem)

        # serialize the pipeline to a string for storage in db if one is provided
        search_template: str = None
        if message.HasField('template'):
            search_template_obj = api_utils.decode_pipeline_description(message.template, pipeline.Resolver())
            search_template = search_template_obj.to_json()

        # Create search row in DB
        search = models.Searches(id=search_id)
        self.session.add(search)
        self.session.commit()

        prob = json.dumps(prob)

        task = models.Tasks(problem=prob,
                            pipeline=search_template,
                            type="EXLINE",
                            dataset_uri=dataset_uri,
                            id=self._generate_id(),
                            search_id=search_id)
        self.session.add(task)

        # Add all to DB
        self.session.commit()
        return search_id

    def _is_search_ended(self, search_id):
        unended_validate_tasks = self.session.query(models.Tasks) \
                                               .filter(models.Tasks.search_id==str(search_id)) \
                                               .filter(models.Tasks.type=="VALIDATE") \
                                               .filter(models.Tasks.ended==False) \
                                               .all()
        return not unended_validate_tasks

    def GetSearchSolutionsResults(self, request):
        """
        Searches for correctly exited EXLINE tasks
        associated with given search id
        """
        search_id = self.validator.validate_get_search_solutions_results_request(request, self.session)
        seen_ids = []

        # Verifies successful 'tasks' one by one
        while True:
            task = self.session.query(models.Tasks) \
                               .filter(models.Tasks.search_id==str(search_id)) \
                               .filter(models.Tasks.type=="EXLINE") \
                               .filter(models.Tasks.ended==True) \
                               .filter(models.Tasks.error==False) \
                               .filter(~models.Tasks.id.in_(seen_ids)) \
                               .first()

            if task:
                self.session.refresh(task)
                # Add id to seen_ids so don't return again
                seen_ids.append(task.id)
                # Check if Solution already exists
                solution = self.session.query(models.Solutions) \
                                       .filter(models.Solutions.search_id==search_id) \
                                       .filter(models.Solutions.task_id==task.id) \
                                       .first()
                # Generate ValidSolution row if has not
                # been previously verified
                if not solution:

                    # Link the task to the solution
                    solution_id = task.id

                    solution = models.Solutions(
                        id=solution_id,
                        search_id=search_id,
                        task_id=task.id)
                    self.session.add(solution)

                # End session
                self.session.commit()

                search_ended = self._is_search_ended(search_id)
                if search_ended:
                    progress = "COMPLETED"
                    progress_msg = self.msg.make_progress_msg(progress)
                    yield self.msg.make_get_search_solutions_result(solution_id, progress_msg)
                    break
                if not search_ended:
                    progress = "RUNNING"
                    progress_msg = self.msg.make_progress_msg(progress)
                    yield self.msg.make_get_search_solutions_result(solution_id, progress_msg)
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
                                score_config_id=conf_id)
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
        # Validate request is in required format, extract if it is
        solution_id = self.validator.validate_fit_solution_request(message)

        # Create request row in DB
        request_id = self._generate_id()
        request = models.Requests(id=request_id,
                                  solution_id=solution_id)
        self.session.add(request)

        # Get solution DAG for task
        solution, task = self.session.query(models.Solutions,models.Tasks) \
                         .filter(models.Solutions.id==solution_id) \
                         .filter(models.Solutions.task_id==models.Tasks.id) \
                         .first()

        # If task has already fit on same dataset_uri,
        # do not create the task
        dataset_uri = self.msg.get_dataset_uri(message)
        if not dataset_uri or dataset_uri == task.dataset_uri:
            # Create the fit_solution here
            fit_solution_id = self._generate_id()
            fit_solution = models.FitSolution(
                id=fit_solution_id,
                solution_id=solution_id,
                task_id=task.id)
            self.session.add(fit_solution)
            # Link the request to the fit_solution
            request.fit_solution_id = fit_solution_id
            self.session.commit()
            return request_id
        else:
            # Create the task otherwise
            task = models.Tasks(id=self._generate_id(),
                                DAG=task.DAG,
                                type="FIT",
                                solution_id=solution_id,
                                fit_solution_id=fit_solution_id,
                                dataset_uri=dataset_uri,
                                request_id=request_id)
            self.session.add(task)

        # Add all to DB
        self.session.commit()

        return request_id

    def GetFitSolutionResults(self, message):
        request_id = self.validator.validate_get_fit_solution_results_request(message, self.session)

        while True:
            task = self.session.query(models.Tasks) \
                               .filter(models.Tasks.request_id==request_id) \
                               .first()

            # If the solution has already been fit on the same dataset_uri
            # there will be no task for it, get info from request instead
            if task is None:
                request = self.session.query(models.Requests) \
                      .filter(models.Requests.id==request_id) \
                      .first()
                fit_solution_id = getattr(request, 'fit_solution_id', False)
                task_complete = True if fit_solution_id else False
            # otherwise get the necessary info from the FIT task
            else:
                task_complete = task.ended
                fit_solution_id = task.fit_solution_id
                self.session.refresh(task)

            # Ensure task is reloaded on next access
            if task != None:
                self.session.expire(task)

            if not task_complete:
                self.logger.debug("FIT task not complete, waiting")
                yield False
            if task_complete:
                progress_msg = self.msg.make_progress_msg("COMPLETED")
                yield self.msg.make_get_fit_solution_results_response(fit_solution_id, progress_msg)
                break

    def ProduceSolution(self, message):
        # Generate request ID
        request_id = self._generate_id()

        # Validate request is in required format, extract if it is
        extracted_fields = self.validator.validate_produce_solution_request(message)
        fitted_solution_id, dataset_uri, output_key = extracted_fields

        # Get existing fit_solution.id
        fit_solution = self.session.query(models.FitSolution) \
                          .filter(models.FitSolution.id==fitted_solution_id) \
                          .first()

        # .first() query returns either first row from query or None
        if fit_solution is None:
            raise ValueError("Fitted solution id {} doesn't exist".format(fitted_solution_id))

        # add a produce task to the tasks table
        task_id = self._generate_id()
        task = models.Tasks(id=task_id,
                            type="PRODUCE",
                            request_id=request_id,
                            fit_solution_id=fit_solution.task_id,
                            dataset_uri=dataset_uri,
                            output_key=output_key)
        self.session.add(task)
        self.session.commit()

        # make a record for the request and commit to the database
        request_record = models.Requests(id=request_id,
                                         task_id=task.id,
                                         type="PRODUCE",
                                         fit_solution_id=fit_solution.id)
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

                # TODO(jtorrez): predictions filename creation should live somewhere better than utils
                preds_path = utils.make_preds_filename(task.id)

                # check the file actually exists
                if not preds_path.exists() and not preds_path.is_file():
                    raise FileNotFoundError("Predictions file {} doesn't exist".format(preds_path))

                progress_msg = self.msg.make_progress_msg("COMPLETED")
                yield self.msg.make_get_produce_solution_results_response(preds_path, task.output_key, progress_msg)
                break


    def DescribeSolution(self, request):
        # Validate the solution_id
        solution_id = self.validator.validate_describe_solution_request(request,
                                                                        self.session)

        # Get the task that ran with the solution
        solution, task = self.session.query(models.Solutions,models.Tasks) \
                                     .filter(models.Solutions.id==solution_id) \
                                     .filter(models.Solutions.task_id==models.Tasks.id) \
                                     .first()

        dag = task.DAG

        return json.loads(dag)

    def SolutionExport(self, request):
        """Output pipeline JSON and "executeable" for D3M evaluation.

        NOTE: This method is HIGHLY SPECIFIC to the eval and would be WONTFIX if not required
        for the eval. You will notice the output folders are hardcoded, this is a known and intentional
        limitation.
        """
        solution_id, rank = self.validator.validate_solution_export_request(request)
        #solution_id = self.validator.validate_fitted_solution_id_exists(fitted_soln_id, self.session, request)

        """
        fit_solution, task = self.session.query(models.FitSolution, models.Tasks) \
                                         .filter(models.FitSolution.id==fitted_soln_id) \
                                         .filter(models.FitSolution.task_id==models.Tasks.id) \
                                         .first()
        """
        solution, task = self.session.query(models.Solutions, models.Tasks) \
                                         .filter(models.Solutions.id==solution_id) \
                                         .filter(models.Solutions.task_id==models.Tasks.id) \
                                         .first()

        export(task, rank)
