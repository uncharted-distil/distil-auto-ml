from google.protobuf import json_format

import models
from server.messages import Messaging

class RequestValidator:
    def __init__(self):
        self.msg = Messaging()

    def _validate_solution_id_exists(self, solution_id, session):
        solution_id = str(solution_id)
        solution_id_record = session.query(models.Solutions) \
                                    .filter(models.Solutions.id==solution_id) \
                                    .first()

        if solution_id_record is None:
            raise ValueError("Solution ID doesn't exist: {}".format(request))

        return solution_id

    def _validate_request_exists(self, request, session):
        request_id = self.msg.get_request_id(request)

        request_record = session.query(models.Requests) \
                                .filter(models.Requests.id==request_id) \
                                .first()

        if request_record is None:
            raise ValueError("Request ID doesn't exist: {}".format(request))

        return request_id

    def validate_fitted_solution_id_exists(self, fitted_soln_id, session):
        fitted_solution_id_record = session.query(models.FitSolution) \
                                    .filter(models.FitSolution.id==fitted_soln_id) \
                                    .first()

        if fitted_solution_id_record is None:
            raise ValueError("FittedSolution ID doesn't exist: {}".format(request))

        return fitted_soln_id


    def validate_search_solutions_request(self, request):
        dataset_uri = self.msg.get_dataset_uri(request)
        if not dataset_uri:
            raise ValueError("Must pass a dataset_uri {}".format(request))

        problem_type = self.msg.get_problem_type(request)
        template = self.msg.get_search_template(request)
        template_dict = self.msg.dump_solution_template(template)
        placeholder_present = False
        if template_dict:
            step_types = [list(s.keys()).pop() for s in template_dict['steps']]
            if 'placeholder' in step_types:
                placeholder_present = True
                if step_types.index('placeholder') != len(step_types) - 1:
                    raise ValueError('Unsupported placeholder position {}'.format(request))
        # This is ok
        if not problem_type and template_dict and not placeholder_present:
            pass
        # This is not ok
        if not problem_type and placeholder_present:
            raise ValueError("Must pass a problem_type with placeholder in template: {}".format(request))
        # Also not ok
        if not problem_type and not template_dict:
            raise ValueError("Must pass a problem_type: {}".format(request))
        return dataset_uri, problem_type

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
        return solution_id

    def validate_produce_solution_request(self, request):
        fitted_solution_id, dataset_uri, output_key = self.msg.unpack_produce_solution_request(request)
        if not fitted_solution_id:
            raise ValueError("Must pass a solution_id: {}".format(request))
        if not dataset_uri:
            raise ValueError("Must pass a dataset_uri {}".format(request))
        if output_key is None:
            raise ValueError("Must specify outputs {}".format(request))
        return fitted_solution_id, dataset_uri, output_key

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
        solution_id = self._validate_solution_id_exists(solution_id, session)

        return solution_id

    def validate_solution_export_request(self, request):
        fitted_solution_id = self.msg.get_fitted_solution_id(request)
        rank = self.msg.get_rank(request)

        if not fitted_solution_id:
            raise ValueError("Must pass a fitted solution_id: {}".format(request))
        # default value of gRPC double field is 0
        if rank == 0:
            raise ValueError("Must specify a rank and rank must be >0: {}".format(request))

        return fitted_solution_id, rank

    def validate_get_fit_solution_results_request(self, request, session):
        return self._validate_request_exists(request, session)

    def validate_get_score_solution_results_request(self, request, session):
        return self._validate_request_exists(request, session)

    def validate_get_produce_solution_results_request(self, request, session):
        return self._validate_request_exists(request, session)
