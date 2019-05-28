import grpc
import time
import logging

from concurrent import futures
from multiprocessing import pool

from api import core_pb2, core_pb2_grpc

from server.messages import Messaging
from server.task_manager import TaskManager


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def _unary_unary_interceptor(servicer, method_name, context, request):
    """TaskManager wrapper for simple RPC single request and response methods.

    Calls method on manager and handles proper gRPC error raising and status codes.

    As more exceptions are added that have specific meaning, they can be added below.
    """
    manager = TaskManager()
    try:
        func = getattr(manager, method_name)
        result = func(request)
    except ValueError as e:
        servicer.logger.info("Request not properly formatted, aborting gRPC call: {}".format(e))
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
    except Exception as e:
        servicer.logger.exception("Unexpected error occured, aborting gRPC call: {}".format(e))
        context.abort(grpc.StatusCode.INTERNAL, "Internal server error occurred")
    finally:
        manager.close()
    return result

def _unary_stream_interceptor(servicer, method_name, context, request):
    """TaskManager wrapper for response-streaming RPC methods.

    Calls method on manager and handles proper gRPC error raising and status codes.

    Interceptor is a generator, can be directly `yield from`d

    As more exceptions are added that have specific meaning, they can be added below.
    """
    manager = TaskManager()
    try:
        func = getattr(manager, method_name)
        for message in func(request):
            if message:
                servicer.logger.debug("{}Response: {}".format(method_name, message))
                yield message
            if not message:
                # yield in_progress message?
                time.sleep(3)
    except ValueError as e:
        servicer.logger.info("Request not properly formatted, aborting gRPC call: {}".format(e))
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
    except Exception as e:
        servicer.logger.exception("Unexpected error occured, aborting gRPC call: {}".format(e))
        context.abort(grpc.StatusCode.INTERNAL, "Internal server error occurred")
    finally:
        manager.close()

class ServerServicer(core_pb2_grpc.CoreServicer):
    """
    * NO direct DB access happens here - all in task_manager
    """

    UNIMPLEMENTED_MSG = "Ah, ah, ah, you didn't say the magic word"

    def __init__(self):
        self.logger = logging.getLogger('distil.server.ServerServicer')
        self.logger.info('Initialized Distil ServerServicer')
        self.msg = Messaging()

    def SearchSolutions(self, request, context):
        """
        Creates 'tasks' to be run by main loop.
        """
        self.logger.debug("SearchSolutions: {}".format(request))
        search_id = _unary_unary_interceptor(self, 'SearchSolutions', context, request)
        return self.msg.search_solutions_response(search_id)

    def GetSearchSolutionsResults(self, request, context):
        """
        Pulls messages out of table related to finding a solution
        * calling this _moves_ valid tasks into ValidSolutions
        """
        self.logger.debug("GetSearchSolutionsResults: {}".format(request))
        yield from _unary_stream_interceptor(self, 'GetSearchSolutionsResults', context, request)

    def EndSearchSolutions(self, request, context):
        # WONTFIX
        return core_pb2.EndSearchSolutionsResponse()

    def StopSearchSolutions(self, request, context):
        # WONTFIX
        return core_pb2.StopSearchSolutionsResponse()

    def DescribeSolution(self, request, context):
        self.logger.debug("DescribeSolution: {}".format(request))
        solution_description = _unary_unary_interceptor(self, 'DescribeSolution', context, request)
        return self.msg.make_describe_solution_response(solution_description)

    def ScoreSolution(self, request, context):
        self.logger.debug("ScoreSolution: {}".format(request))
        score_request_id = _unary_unary_interceptor(self, 'ScoreSolution', context, request)
        return self.msg.score_solution_response(score_request_id)

    def GetScoreSolutionResults(self, request, context):
        self.logger.debug("GetScoreSolutionResults: {}".format(request))
        yield from _unary_stream_interceptor(self, 'GetScoreSolutionResults', context, request)

    def FitSolution(self, request, context):
        self.logger.debug("FitSolution: {}".format(request))
        request_id = _unary_unary_interceptor(self, 'FitSolution', context, request)
        return self.msg.make_fit_solution_message(request_id)

    def GetFitSolutionResults(self, request, context):
        self.logger.debug("GetFitSolutionResults: {}".format(request))
        yield from _unary_stream_interceptor(self, 'GetFitSolutionResults', context, request)

    def ProduceSolution(self, request, context):
        self.logger.debug("ProduceSolution: {}".format(request))
        request_id = _unary_unary_interceptor(self, 'ProduceSolution', context, request)
        return self.msg.make_produce_solution_response(request_id)

    def GetProduceSolutionResults(self, request, context):
        self.logger.debug("GetProduceSolutionResults: {}".format(request))
        yield from _unary_stream_interceptor(self, 'GetProduceSolutionResults', context, request)

    def SolutionExport(self, request, context):
        self.logger.debug("SolutionExport: {}".format(request))
        # TaskManager.SolutionExport returns nothing because SolutionExportResponse is empty
        _ = _unary_unary_interceptor(self, 'SolutionExport', context, request)
        return self.msg.make_solution_export_response()

    def UpdateProblem(self, request, context):
        # WONTFIX
        context.abort(grpc.StatusCode.UNIMPLEMENTED, self.UNIMPLEMENTED_MSG)

    def ListPrimitives(self, request, context):
        # TODO
        context.abort(grpc.StatusCode.UNIMPLEMENTED, self.UNIMPLEMENTED_MSG)

    def Hello(self, request, context):
        return self.msg.make_hello_response_message()


class Server():
    def __init__(self):
        self.logger = logging.getLogger('xline.server.Server')
        self.logger.info('Initializing xline gRPC server')
        self.server_thread = pool.ThreadPool(processes=1,)

    def create_server(self, port):
        self.servicer = ServerServicer()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        core_pb2_grpc.add_CoreServicer_to_server(
            self.servicer, self.server)
        address = '[::]:{}'.format(port)
        self.server.add_insecure_port(address)
        self.logger.info("Starting server at {}".format(address))
        self.server.start()
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
        self.server.stop(0)

    def start(self, port):
        self.server_thread.apply_async(self.create_server, (port,))
        self.logger.info('Started gRPC server, listening for requests')
