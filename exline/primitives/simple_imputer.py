import os

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from preprocessing.utils import MISSING_VALUE_INDICATOR


__all__ = ('SimpleImputerPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )


class SimpleImputerPrimitive(transformer.TransformerPrimitiveBase[container.ndarray, container.ndarray, Hyperparams]):
    """
    A primitive that imputes simples.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'a9bbd6ba-d349-46ce-90b6-541db90236ee',
            'version': '0.1.0',
            'name': "Simple imputer",
            'python_path': 'd3m.primitives.data_transformation.imputer.ExlineSimpleImputer',
            'source': {
                'name': 'exline',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/cdbethune/d3m-exline/primitives/simple_imputer.py',
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

    def produce(self, *, inputs: container.ndarray, timeout: float = None, iterations: int = None) -> base.CallResult[container.ndarray]:
        cols = self.hyperparams['use_columns']
        numerical_inputs = inputs[:,cols]
        imputer = SimpleImputer()
        imputer.fit(numerical_inputs)
        result = container.ndarray(imputer.transform(inputs))
        return base.CallResult(result)
