import os
import typing

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('ExlineGraphLoaderPrimitive',)

Inputs = container.Dataset
Outputs = container.DataFrame

import logging
import networkx as nx

logger = logging.getLogger(__name__)



class Hyperparams(hyperparams.Hyperparams):
    dataframe_resource = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description=".",
    )


class ExlineGraphLoaderPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which loads.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'ae0797506-ea7b-4a7f-a7e4-2f91e2082f05',
            'version': '0.1.0',
            'name': "Load graphs into a parseable object",
            'python_path': 'd3m.primitives.data_transformation.load_graphs.ExlineGraphLoader',
            'source': {
                'name': 'exline',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/cdbethune/d3m-exline/primitives/load_graphs.py',
                    'https://github.com/cdbethune/d3m-exline',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/cdbethune/d3m-exline.git@{git_commit}#egg=d3m-exline'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        dataframe_resource_id, dataframe = base_utils.get_tabular_resource(inputs, self.hyperparams['dataframe_resource'])

        graph1 = inputs['0']
        int2str_map = dict(zip(graph1.nodes, [str(n) for n in graph1.nodes]))
        graph1 = nx.relabel_nodes(graph1, mapping=int2str_map)

        graph2 = inputs['1']
        int2str_map = dict(zip(graph2.nodes, [str(n) for n in graph2.nodes]))
        graph2 = nx.relabel_nodes(graph2, mapping=int2str_map)

        dataframe.metadata = self._update_metadata(inputs.metadata, dataframe_resource_id)

        assert isinstance(dataframe, container.DataFrame), type(dataframe)

        return base.CallResult([dataframe, graph1, graph2])

    @classmethod
    def _update_metadata(cls, metadata: metadata_base.DataMetadata, resource_id: metadata_base.SelectorSegment) -> metadata_base.DataMetadata:
        resource_metadata = dict(metadata.query((resource_id,)))

        if 'structural_type' not in resource_metadata or not issubclass(resource_metadata['structural_type'], container.DataFrame):
            raise TypeError("The Dataset resource is not a DataFrame, but \"{type}\".".format(
                type=resource_metadata.get('structural_type', None),
            ))

        resource_metadata.update(
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            },
        )

        new_metadata = metadata_base.DataMetadata(resource_metadata)

        new_metadata = metadata.copy_to(new_metadata, (resource_id,))

        # Resource is not anymore an entry point.
        new_metadata = new_metadata.remove_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint')

        return new_metadata

    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_base.Metadata, type]],
                   hyperparams: Hyperparams) -> typing.Optional[metadata_base.DataMetadata]:
        output_metadata = super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)

        # If structural types didn't match, don't bother.
        if output_metadata is None:
            return None

        if method_name != 'produce':
            return output_metadata

        if 'inputs' not in arguments:
            return output_metadata

        inputs_metadata = typing.cast(metadata_base.DataMetadata, arguments['inputs'])

        dataframe_resource_id = base_utils.get_tabular_resource_metadata(inputs_metadata, hyperparams['dataframe_resource'])

        return cls._update_metadata(inputs_metadata, dataframe_resource_id)
