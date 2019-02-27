import os
from typing import List
from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import pandas as pd

__all__ = ('SimpleColumnParserPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    column_types = hyperparams.Hyperparameter[List[type]](
        default=[],
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Schema type of each column in order they stored in the input dataframe",
    )

class SimpleColumnParserPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFrame, Hyperparams]):
    """
    A primitive that imputes simples.
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
        print('>> SIMPLE COLUMN PARSER START')
        # retype from object to int, float, bool or string
        column_types = self.hyperparams['column_types']
        for i, col_type in enumerate(column_types):
            if col_type is int or col_type is float:
                inputs.iloc[:,i] = pd.to_numeric(inputs.iloc[:,i])
            elif col_type is bool:
                inputs.iloc[:,i] = inputs.iloc[:,i].astype('bool')
        # flip the d3mIndex to be the df index
        inputs = inputs.set_index('d3mIndex')

        print(inputs)
        print(inputs.dtypes)
        print('<< SIMPLE COLUMN PARSER END')
        return base.CallResult(inputs)
