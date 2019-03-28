import os
import logging
from typing import List

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, unsupervised_learning

import pandas as pd
import numpy as np

from exline.preprocessing.transformers import BinaryEncoder


__all__ = ('BinaryEncoderPrimitive',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )

    min_binary = hyperparams.Hyperparameter[int](
        default=17,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Min number of labels for binary encoding",
    )

class Params(params.Params):
    pass

class BinaryEncoderPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
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
                    'https://github.com/cdbethune/d3m-exline/exline/primitives/binary_encoder.py',
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

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:
        base.PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed)

    def __getstate__(self) -> dict:
        state = base.PrimitiveBase.__getstate__(self)
        state['models'] = self._encoders
        state['colums'] = self._cols
        return state

    def __setstate__(self, state: dict) -> None:
        base.PrimitiveBase.__setstate__(self, state)
        self._encoders = state['models']
        self._cols = state['columns']

    def set_training_data(self, *, inputs: container.DataFrame) -> None:
        self._inputs = inputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        logger.debug('Fitting binary encoder')

        cols = list(self.hyperparams['use_columns'])

        if cols is None or len(cols) is 0:
            cols = []
            for idx, c in enumerate(self._inputs.columns):
                if self._inputs[c].dtype == object:
                    num_labels = len(set(self._inputs[c]))
                    if num_labels > 1 and num_labels >= self.hyperparams['min_binary'] and not self._detect_text(self._inputs[c]):
                        cols.append(idx)

        logger.debug(f'Found {len(cols)} columns to encode')

        self._cols = cols
        self._encoders: List[BinaryEncoder] = []
        if len(cols) is 0:
            return base.CallResult(None)

        # add the binary encoded columns and remove the source
        for i, c in enumerate(cols):
            encoder = BinaryEncoder()
            categorical_inputs = self._inputs.iloc[:,c]
            encoder.fit(categorical_inputs)
            self._encoders.append(encoder)

        return base.CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        if len(self._cols) == 0:
            return base.CallResult(inputs)

        # add the binary encoded columns and remove the source
        outputs = inputs.copy()
        for i, c in enumerate(self._cols):
            categorical_inputs = outputs.iloc[:,c]
            result = self._encoders[i].transform(categorical_inputs)
            for j in range(result.shape[1]):
                outputs[(f'__binary_{i * result.shape[1] + j}')] = result[:,j]

        outputs.drop(outputs.columns[self._cols], axis=1, inplace=True)

        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return

    @classmethod
    def _detect_text(cls, X: container.DataFrame, thresh: int = 8) -> bool:
        """ returns true if median entry has more than `thresh` tokens"""
        X = X[X.notnull()]
        n_toks = X.apply(lambda xx: len(str(xx).split(' '))).values
        return np.median(n_toks) >= thresh