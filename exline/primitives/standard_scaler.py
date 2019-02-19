import os

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from preprocessing.utils import MISSING_VALUE_INDICATOR


__all__ = ('StandardScalerPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )


class StandardScalerPrimitive(transformer.TransformerPrimitiveBase[container.ndarray, container.ndarray, Hyperparams]):
    """
    A primitive that scales standards.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'c71596b2-1ca7-4a16-8f1c-f6affab31cfb',
            'version': '0.1.0',
            'name': "Standard scaler",
            'python_path': 'd3m.primitives.data_transformation.standard_scaler.ExlineStandardScaler',
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
        scaler = StandardScaler()
        scaler.fit(numerical_inputs)
        result = container.ndarray(scaler.transform(inputs))
        return base.CallResult(result)
