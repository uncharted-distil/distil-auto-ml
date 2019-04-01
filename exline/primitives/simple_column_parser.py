import os
import io
from typing import List, Sequence
import logging

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from common_primitives import utils as common_utils

import pandas as pd

__all__ = ('SimpleColumnParserPrimitive',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    pass

class SimpleColumnParserPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFrame, Hyperparams]):
    """
    A primitive that parses simple columns.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '7b67eef9-f14e-4219-bf0c-5222880eac78',
            'version': '0.1.0',
            'name': "Simple column parser",
            'python_path': 'd3m.primitives.data_transformation.column_parser.ExlineSimpleColumnParser',
            'source': {
                'name': 'exline',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/cdbethune/d3m-exline/primitives/simple_column_parser.py',
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
                metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:

        logger.debug(f'Running {__name__} produce')

        outputs = inputs.copy()

        num_cols = outputs.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
        remove_indices = []
        for i in range(num_cols):
            semantic_types = outputs.metadata.query((metadata_base.ALL_ELEMENTS,i))['semantic_types']
            # mark target + index for removal
            if 'https://metadata.datadrivendiscovery.org/types/Target' in semantic_types or \
                'https://metadata.datadrivendiscovery.org/types/TrueTarget' in semantic_types or \
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in semantic_types:
                remove_indices.append(i)

            # update the structural / df type from the semantic type
            outputs = self._update_type_info(semantic_types, outputs, i)

        # flip the d3mIndex to be the df index as well
        outputs = outputs.set_index('d3mIndex', drop=False)

        # remove target and primary key
        outputs = common_utils.remove_columns(outputs, remove_indices)

        logger.debug(f'\n{outputs.dtypes}')
        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)

    def produce_target(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Running {__name__} produce_target')

        outputs = inputs.copy()

        # find the target column and remove all others
        num_cols = outputs.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
        target_idx = -1
        for i in range(num_cols):
            semantic_types = outputs.metadata.query((metadata_base.ALL_ELEMENTS,i))['semantic_types']
            if 'https://metadata.datadrivendiscovery.org/types/Target' in semantic_types or \
               'https://metadata.datadrivendiscovery.org/types/TrueTarget' in semantic_types:
                target_idx = i
                outputs = self._update_type_info(semantic_types, outputs, i)
            elif 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in semantic_types:
                outputs = self._update_type_info(semantic_types, outputs, i)

        # flip the d3mIndex to be the df index as well
        outputs = outputs.set_index('d3mIndex', drop=False)

        remove_indices = set(range(num_cols))
        remove_indices.remove(target_idx)
        outputs = common_utils.remove_columns(outputs, remove_indices)

        logger.debug(f'\n{outputs.dtypes}')
        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)

    @classmethod
    def _update_type_info(self, semantic_types: Sequence[str], outputs: container.DataFrame, i: int) -> container.DataFrame:
        # update the structural / df type from the semantic type
        if 'http://schema.org/Integer' in semantic_types:
            outputs.metadata = outputs.metadata.update_column(i, {'structural_type': int})
            outputs.iloc[:,i] = pd.to_numeric(outputs.iloc[:,i])
        elif 'http://schema.org/Float' in semantic_types:
            outputs.metadata = outputs.metadata.update_column(i, {'structural_type': float})
            outputs.iloc[:,i] = pd.to_numeric(outputs.iloc[:,i])
        elif 'http://schema.org/Boolean' in semantic_types:
            outputs.metadata = outputs.metadata.update_column(i, {'structural_type': bool})
            outputs.iloc[:,i] = outputs.iloc[:,i].astype('bool')

        return outputs