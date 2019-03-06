import os
from typing import List
from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import pandas as pd

__all__ = ('SimpleColumnParserPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    target = hyperparams.Hyperparameter[str](
        default='',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Name of target column"
    )
    column_types = hyperparams.Hyperparameter[List[type]](
        default=[],
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Schema type of each column in order they stored in the input dataframe",
    )

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
        print('>> SIMPLE COLUMN PARSER START')
        outputs = inputs.copy()

        # retype from object to int, float, bool or string
        column_types = self.hyperparams['column_types']
        for i, col_type in enumerate(column_types):
            if col_type is int or col_type is float:
                outputs.iloc[:,i] = pd.to_numeric(outputs.iloc[:,i])
            elif col_type is bool:
                outputs.iloc[:,i] = outputs.iloc[:,i].astype('bool')
        # flip the d3mIndex to be the df index
        outputs = outputs.set_index('d3mIndex')

        # drop the targets
        outputs = outputs.drop(self.hyperparams['target'], axis=1)

        print(outputs)
        print(outputs.dtypes)
        print('<< SIMPLE COLUMN PARSER END')
        #return base.CallResult(outputs)
        return base.CallResult(outputs)

    def produce_target(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        return base.CallResult(inputs[self.hyperparams['target']]) 