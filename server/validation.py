from google.protobuf import json_format

import models
from server.messages import Messaging

import logging


class RequestValidator:
    def __init__(self):
        self.msg = Messaging()
        self.logger = logging.getLogger('distil.TaskManager')

    def _validate_solution_id_exists(self, solution_id, session, request):
        solution_id = str(solution_id)
        solution_id_record = session.query(models.Solutions) \
                                    .filter(models.Solutions.id==solution_id) \
                                    .first()

        if solution_id_record is None:
            raise ValueError("Solution ID does not exist: {}".format(request))

        return solution_id

    def _validate_request_exists(self, request, session):
        request_id = self.msg.get_request_id(request)

        request_record = session.query(models.Requests) \
                                .filter(models.Requests.id==request_id) \
                                .first()

        if request_record is None:
            raise ValueError("Request ID does not exist: {}".format(request))

        return request_id

    def validate_fitted_solution_id_exists(self, soln_id, session, request):
        fit_solution_id_record = session.query(models.FitSolution) \
                                    .filter(models.FitSolution.id==soln_id) \
                                    .first()

        solution_id_record = session.query(models.Solutions) \
                                    .filter(models.Solutions.id==soln_id) \
                                    .first()

        if fit_solution_id_record is None and solution_id_record is None:
            raise ValueError("Provided ID does not exist: {}".format(request))

        return fitted_solution_id_record


    def validate_search_solutions_request(self, request):
        dataset_uri = self.msg.get_dataset_uri(request)
        if not dataset_uri:
            raise ValueError("Must pass a dataset_uri {}".format(request))

        # if the problem contents are empty set the returned problem to None
        problem_type = self.msg.get_problem_type(request)
        problem = request.problem if problem_type else None

        # if the template contents are empty, set the returned template to None
        template = self.msg.get_search_template(request)
        template_dict = self.msg.dump_solution_template(template)
        if len(template_dict) == 0:
            template = None
        placeholder_present = False

        if template_dict:
            step_types = [list(s.keys()).pop() for s in template_dict['steps']]
            if 'placeholder' in step_types:
                placeholder_present = True
            # Placeholder doesn't have to be last primitive in step list, it just has to be the
            # the single output from the pipeline DAG.  Assumption below doesn't hold in that case.
            #     if step_types.index('placeholder') != len(step_types) - 1:
            #         raise ValueError('Unsupported placeholder position {}'.format(request))
        # This is ok
        if not problem_type and template_dict and not placeholder_present:
            pass
        # This is not ok
        if not problem_type and placeholder_present:
            raise ValueError("Must pass a problem_type with placeholder in template: {}".format(request))
        # Also not ok
        if not problem_type and not template_dict:
            raise ValueError("Must pass a problem_type: {}".format(request))
        return dataset_uri, problem, template

    def validate_score_solution_request(self, request):
        r = self.msg.unpack_score_solution_request(request)
        # unpack fields to be validated from request
        solution_id, _, metrics, method, _, _, _, _, _ = r

        if not solution_id:
            raise ValueError("Must pass a solution_id: {}".format(request))
        if not metrics:
            raise ValueError("Must pass at least one scoring metric: {}".format(request))
        if not method:
            raise ValueError("Must pass scoring configuration with evaluation method: {}".format(request))
        return r

    def validate_fit_solution_request(self, request):
        solution_id = self.msg.get_fit_solution_id(request)
        if not solution_id:
            raise ValueError("Must pass a solution_id: {}".format(request))
        dataset_uri = self.msg.get_dataset_uri(request)
        if not dataset_uri:
            raise ValueError("Must pass a dataset URI: {}".format(request))
        return solution_id, dataset_uri

    def validate_produce_solution_request(self, request):
        fitted_solution_id, dataset_uri, output_keys, output_types = self.msg.unpack_produce_solution_request(request)
        if not fitted_solution_id:
            raise ValueError("Must pass a solution_id: {}".format(request))
        if not dataset_uri:
            raise ValueError("Must pass a dataset_uri {}".format(request))
        if output_keys is None:
            raise ValueError("Must specify outputs {}".format(request))
        # note that output_types can be empty based on the ta3ta2 api spec so we don't
        # validate it here
        return fitted_solution_id, dataset_uri, output_keys, output_types

    def validate_get_search_solutions_results_request(self, request, session):
        search_id = str(self.msg.get_search_id(request))

        search = session.query(models.Searches) \
                        .filter(models.Searches.id==search_id) \
                        .first()

        if search is None:
            raise ValueError("Search ID doesn't exist: {}".format(request))

        return search_id

    def validate_describe_solution_request(self, request, session):
        solution_id = self.msg.get_solution_id(request)
        solution_id = self._validate_solution_id_exists(solution_id, session, request)

        return solution_id

    def validate_solution_export_request(self, request):
        solution_id = self.msg.get_solution_id(request)
        rank = self.msg.get_rank(request)

        if not solution_id:
            raise ValueError("Must pass a solution_id: {}".format(request))
        # default value of gRPC double field is 0
        if rank < 0:
            raise ValueError("Must specify a rank and rank must be >=0: {}".format(request))

        return solution_id, rank

    def validate_save_solution_request(self, request):
        solution_id = self.msg.get_solution_id(request)

        if not solution_id:
            raise ValueError("Must pass a solution_id: {}".format(request))

        return solution_id

    def validate_save_fitted_solution_request(self, request):
        fitted_solution_id = self.msg.get_fitted_solution_id(request)

        if not fitted_solution_id:
            raise ValueError("Must pass a fitted_solution_id: {}".format(request))

        return fitted_solution_id

    def validate_load_solution_request(self, request):
        solution_uri = self.msg.get_solution_uri(request)

        if not solution_uri:
            raise ValueError("Must pass a solution_uri: {}".format(request))

        return solution_uri

    def validate_load_fitted_solution_request(self, request):
        fitted_solution_uri = self.msg.get_fitted_solution_uri(request)

        if not fitted_solution_uri:
            raise ValueError("Must pass a fitted_solution_uri: {}".format(request))

        return fitted_solution_uri

    def validate_get_fit_solution_results_request(self, request, session):
        return self._validate_request_exists(request, session)

    def validate_get_score_solution_results_request(self, request, session):
        return self._validate_request_exists(request, session)

    def validate_get_produce_solution_results_request(self, request, session):
        return self._validate_request_exists(request, session)
