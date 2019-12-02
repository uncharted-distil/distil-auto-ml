# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from . import core_pb2 as core__pb2


class CoreStub(object):
  """See each message's comments for information about each particular call.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.SearchSolutions = channel.unary_unary(
        '/Core/SearchSolutions',
        request_serializer=core__pb2.SearchSolutionsRequest.SerializeToString,
        response_deserializer=core__pb2.SearchSolutionsResponse.FromString,
        )
    self.GetSearchSolutionsResults = channel.unary_stream(
        '/Core/GetSearchSolutionsResults',
        request_serializer=core__pb2.GetSearchSolutionsResultsRequest.SerializeToString,
        response_deserializer=core__pb2.GetSearchSolutionsResultsResponse.FromString,
        )
    self.EndSearchSolutions = channel.unary_unary(
        '/Core/EndSearchSolutions',
        request_serializer=core__pb2.EndSearchSolutionsRequest.SerializeToString,
        response_deserializer=core__pb2.EndSearchSolutionsResponse.FromString,
        )
    self.StopSearchSolutions = channel.unary_unary(
        '/Core/StopSearchSolutions',
        request_serializer=core__pb2.StopSearchSolutionsRequest.SerializeToString,
        response_deserializer=core__pb2.StopSearchSolutionsResponse.FromString,
        )
    self.DescribeSolution = channel.unary_unary(
        '/Core/DescribeSolution',
        request_serializer=core__pb2.DescribeSolutionRequest.SerializeToString,
        response_deserializer=core__pb2.DescribeSolutionResponse.FromString,
        )
    self.ScoreSolution = channel.unary_unary(
        '/Core/ScoreSolution',
        request_serializer=core__pb2.ScoreSolutionRequest.SerializeToString,
        response_deserializer=core__pb2.ScoreSolutionResponse.FromString,
        )
    self.GetScoreSolutionResults = channel.unary_stream(
        '/Core/GetScoreSolutionResults',
        request_serializer=core__pb2.GetScoreSolutionResultsRequest.SerializeToString,
        response_deserializer=core__pb2.GetScoreSolutionResultsResponse.FromString,
        )
    self.FitSolution = channel.unary_unary(
        '/Core/FitSolution',
        request_serializer=core__pb2.FitSolutionRequest.SerializeToString,
        response_deserializer=core__pb2.FitSolutionResponse.FromString,
        )
    self.GetFitSolutionResults = channel.unary_stream(
        '/Core/GetFitSolutionResults',
        request_serializer=core__pb2.GetFitSolutionResultsRequest.SerializeToString,
        response_deserializer=core__pb2.GetFitSolutionResultsResponse.FromString,
        )
    self.ProduceSolution = channel.unary_unary(
        '/Core/ProduceSolution',
        request_serializer=core__pb2.ProduceSolutionRequest.SerializeToString,
        response_deserializer=core__pb2.ProduceSolutionResponse.FromString,
        )
    self.GetProduceSolutionResults = channel.unary_stream(
        '/Core/GetProduceSolutionResults',
        request_serializer=core__pb2.GetProduceSolutionResultsRequest.SerializeToString,
        response_deserializer=core__pb2.GetProduceSolutionResultsResponse.FromString,
        )
    self.SolutionExport = channel.unary_unary(
        '/Core/SolutionExport',
        request_serializer=core__pb2.SolutionExportRequest.SerializeToString,
        response_deserializer=core__pb2.SolutionExportResponse.FromString,
        )
    self.DataAvailable = channel.unary_unary(
        '/Core/DataAvailable',
        request_serializer=core__pb2.DataAvailableRequest.SerializeToString,
        response_deserializer=core__pb2.DataAvailableResponse.FromString,
        )
    self.ListPrimitives = channel.unary_unary(
        '/Core/ListPrimitives',
        request_serializer=core__pb2.ListPrimitivesRequest.SerializeToString,
        response_deserializer=core__pb2.ListPrimitivesResponse.FromString,
        )
    self.Hello = channel.unary_unary(
        '/Core/Hello',
        request_serializer=core__pb2.HelloRequest.SerializeToString,
        response_deserializer=core__pb2.HelloResponse.FromString,
        )
    self.SaveSolution = channel.unary_unary(
        '/Core/SaveSolution',
        request_serializer=core__pb2.SaveSolutionRequest.SerializeToString,
        response_deserializer=core__pb2.SaveSolutionResponse.FromString,
        )
    self.LoadSolution = channel.unary_unary(
        '/Core/LoadSolution',
        request_serializer=core__pb2.LoadSolutionRequest.SerializeToString,
        response_deserializer=core__pb2.LoadSolutionResponse.FromString,
        )
    self.SaveFittedSolution = channel.unary_unary(
        '/Core/SaveFittedSolution',
        request_serializer=core__pb2.SaveFittedSolutionRequest.SerializeToString,
        response_deserializer=core__pb2.SaveFittedSolutionResponse.FromString,
        )
    self.LoadFittedSolution = channel.unary_unary(
        '/Core/LoadFittedSolution',
        request_serializer=core__pb2.LoadFittedSolutionRequest.SerializeToString,
        response_deserializer=core__pb2.LoadFittedSolutionResponse.FromString,
        )


class CoreServicer(object):
  """See each message's comments for information about each particular call.
  """

  def SearchSolutions(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetSearchSolutionsResults(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def EndSearchSolutions(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def StopSearchSolutions(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DescribeSolution(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ScoreSolution(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetScoreSolutionResults(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def FitSolution(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetFitSolutionResults(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ProduceSolution(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetProduceSolutionResults(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SolutionExport(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DataAvailable(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ListPrimitives(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Hello(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SaveSolution(self, request, context):
    """Optional.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def LoadSolution(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SaveFittedSolution(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def LoadFittedSolution(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_CoreServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'SearchSolutions': grpc.unary_unary_rpc_method_handler(
          servicer.SearchSolutions,
          request_deserializer=core__pb2.SearchSolutionsRequest.FromString,
          response_serializer=core__pb2.SearchSolutionsResponse.SerializeToString,
      ),
      'GetSearchSolutionsResults': grpc.unary_stream_rpc_method_handler(
          servicer.GetSearchSolutionsResults,
          request_deserializer=core__pb2.GetSearchSolutionsResultsRequest.FromString,
          response_serializer=core__pb2.GetSearchSolutionsResultsResponse.SerializeToString,
      ),
      'EndSearchSolutions': grpc.unary_unary_rpc_method_handler(
          servicer.EndSearchSolutions,
          request_deserializer=core__pb2.EndSearchSolutionsRequest.FromString,
          response_serializer=core__pb2.EndSearchSolutionsResponse.SerializeToString,
      ),
      'StopSearchSolutions': grpc.unary_unary_rpc_method_handler(
          servicer.StopSearchSolutions,
          request_deserializer=core__pb2.StopSearchSolutionsRequest.FromString,
          response_serializer=core__pb2.StopSearchSolutionsResponse.SerializeToString,
      ),
      'DescribeSolution': grpc.unary_unary_rpc_method_handler(
          servicer.DescribeSolution,
          request_deserializer=core__pb2.DescribeSolutionRequest.FromString,
          response_serializer=core__pb2.DescribeSolutionResponse.SerializeToString,
      ),
      'ScoreSolution': grpc.unary_unary_rpc_method_handler(
          servicer.ScoreSolution,
          request_deserializer=core__pb2.ScoreSolutionRequest.FromString,
          response_serializer=core__pb2.ScoreSolutionResponse.SerializeToString,
      ),
      'GetScoreSolutionResults': grpc.unary_stream_rpc_method_handler(
          servicer.GetScoreSolutionResults,
          request_deserializer=core__pb2.GetScoreSolutionResultsRequest.FromString,
          response_serializer=core__pb2.GetScoreSolutionResultsResponse.SerializeToString,
      ),
      'FitSolution': grpc.unary_unary_rpc_method_handler(
          servicer.FitSolution,
          request_deserializer=core__pb2.FitSolutionRequest.FromString,
          response_serializer=core__pb2.FitSolutionResponse.SerializeToString,
      ),
      'GetFitSolutionResults': grpc.unary_stream_rpc_method_handler(
          servicer.GetFitSolutionResults,
          request_deserializer=core__pb2.GetFitSolutionResultsRequest.FromString,
          response_serializer=core__pb2.GetFitSolutionResultsResponse.SerializeToString,
      ),
      'ProduceSolution': grpc.unary_unary_rpc_method_handler(
          servicer.ProduceSolution,
          request_deserializer=core__pb2.ProduceSolutionRequest.FromString,
          response_serializer=core__pb2.ProduceSolutionResponse.SerializeToString,
      ),
      'GetProduceSolutionResults': grpc.unary_stream_rpc_method_handler(
          servicer.GetProduceSolutionResults,
          request_deserializer=core__pb2.GetProduceSolutionResultsRequest.FromString,
          response_serializer=core__pb2.GetProduceSolutionResultsResponse.SerializeToString,
      ),
      'SolutionExport': grpc.unary_unary_rpc_method_handler(
          servicer.SolutionExport,
          request_deserializer=core__pb2.SolutionExportRequest.FromString,
          response_serializer=core__pb2.SolutionExportResponse.SerializeToString,
      ),
      'DataAvailable': grpc.unary_unary_rpc_method_handler(
          servicer.DataAvailable,
          request_deserializer=core__pb2.DataAvailableRequest.FromString,
          response_serializer=core__pb2.DataAvailableResponse.SerializeToString,
      ),
      'ListPrimitives': grpc.unary_unary_rpc_method_handler(
          servicer.ListPrimitives,
          request_deserializer=core__pb2.ListPrimitivesRequest.FromString,
          response_serializer=core__pb2.ListPrimitivesResponse.SerializeToString,
      ),
      'Hello': grpc.unary_unary_rpc_method_handler(
          servicer.Hello,
          request_deserializer=core__pb2.HelloRequest.FromString,
          response_serializer=core__pb2.HelloResponse.SerializeToString,
      ),
      'SaveSolution': grpc.unary_unary_rpc_method_handler(
          servicer.SaveSolution,
          request_deserializer=core__pb2.SaveSolutionRequest.FromString,
          response_serializer=core__pb2.SaveSolutionResponse.SerializeToString,
      ),
      'LoadSolution': grpc.unary_unary_rpc_method_handler(
          servicer.LoadSolution,
          request_deserializer=core__pb2.LoadSolutionRequest.FromString,
          response_serializer=core__pb2.LoadSolutionResponse.SerializeToString,
      ),
      'SaveFittedSolution': grpc.unary_unary_rpc_method_handler(
          servicer.SaveFittedSolution,
          request_deserializer=core__pb2.SaveFittedSolutionRequest.FromString,
          response_serializer=core__pb2.SaveFittedSolutionResponse.SerializeToString,
      ),
      'LoadFittedSolution': grpc.unary_unary_rpc_method_handler(
          servicer.LoadFittedSolution,
          request_deserializer=core__pb2.LoadFittedSolutionRequest.FromString,
          response_serializer=core__pb2.LoadFittedSolutionResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Core', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
