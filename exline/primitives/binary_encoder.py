import os

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import pandas as pd
import numpy as np

from preprocessing.transformers import BinaryEncoder

from preprocessing.utils import MISSING_VALUE_INDICATOR


__all__ = ('BinaryEncoderPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )

class BinaryEncoderPrimitive(transformer.TransformerPrimitiveBase[container.ndarray, container.ndarray, Hyperparams]):
    """
    A primitive that encodes binaries.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'd38e2e28-9b18-4ce4-b07c-9d809cd8b915',
            'version': '0.1.0',
            'name': "Binary encoder",
            'python_path': 'd3m.primitives.data_transformation.encoder.ExlineBinaryEncoder',
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
        categories = self.hyperparams['categories']
        categorical_inputs = inputs[:,cols]
        encoder = BinaryEncoder()
        encoder.fit(categorical_inputs)
        result = container.ndarray(encoder.transform(inputs))
        return base.CallResult(result)