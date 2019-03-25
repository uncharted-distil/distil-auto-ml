import os
from typing import List, Set, Any
import logging

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import compose

logger = logging.getLogger(__name__)

__all__ = ('OneHotEncoderPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )

    max_one_hot = hyperparams.Hyperparameter[int](
        default=16,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Max number of labels for one hot encoding",
    )

class OneHotEncoderPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFrame, Hyperparams]):
    """
    A primitive that encodes one hots.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'd3d421cb-9601-43f0-83d9-91a9c4199a06',
            'version': '0.1.0',
            'name': "One-hot encoder",
            'python_path': 'd3m.primitives.data_transformation.one_hot_encoder.ExlineOneHotEncoder',
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



    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug('Running one hot encoder')

        cols = list(self.hyperparams['use_columns'])

        if cols is None or len(cols) is 0:
            cols = []
            for idx, c in enumerate(inputs.columns):
                if inputs[c].dtype == object:
                    num_labels = len(set(inputs[c]))
                    if num_labels > 1 and num_labels <= self.hyperparams['max_one_hot'] and not self._detect_text(inputs[c]):
                        cols.append(idx)

        logger.debug(f'Found {len(cols)} columns to encode')

        if len(cols) is 0:
            return base.CallResult(inputs)

        input_cols = inputs.iloc[:,cols]

        encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoder.fit(input_cols)

        result = encoder.transform(input_cols)

        # remove the source columns and append the result to the copy
        outputs = inputs.copy()
        outputs.drop(outputs.columns[cols], axis=1, inplace=True)
        for i in range(result.shape[1]):
            outputs[('__onehot_' + str(i))] = result[:,i]


        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)

    @classmethod
    def _detect_text(cls, X: container.DataFrame, thresh: int = 8) -> bool:
        """ returns true if median entry has more than `thresh` tokens"""
        X = X[X.notnull()]
        n_toks = X.apply(lambda xx: len(str(xx).split(' '))).values
        return np.median(n_toks) >= thresh