import os
import logging

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import unsupervised_learning, transformer, base

import pandas as pd
import numpy as np

from sklearn.impute import MissingIndicator

from exline.preprocessing.utils import MISSING_VALUE_INDICATOR

logger = logging.getLogger(__name__)

__all__ = ('MissingIndicatorPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )

class Params(params.Params):
    pass

class MissingIndicatorPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    A primitive that scales standards.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '15587104-0e81-4970-add3-668da63be95b',
            'version': '0.1.0',
            'name': "Standard scaler",
            'python_path': 'd3m.primitives.data_transformation.missing_indicator.ExlineMissingIndicator',
            'source': {
                'name': 'exline',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/cdbethune/d3m-exline/primitives/missing_indicator.py',
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
        state['models'] = self._missing_indicator
        state['colums'] = self._cols
        return state

    def __setstate__(self, state: dict) -> None:
        base.PrimitiveBase.__setstate__(self, state)
        self._missing_indicator = state['models']
        self._cols = state['columns']

    def set_training_data(self, *, inputs: container.DataFrame) -> None:
        self._inputs = inputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        logger.debug(f'Fitting {__name__}')

        # find candidate columns
        cols = list(self.hyperparams['use_columns'])
        if cols is None or len(cols) is 0:
            cols = []
            for idx, c in enumerate(self._inputs.columns):
                if (self._inputs[c].dtype == int or self._inputs[c].dtype == float or self._inputs[c].dtype == bool) and self._inputs[c].isnull().any():
                    cols.append(idx)

        logger.debug(f'Found {len(cols)} cols to process for missing values')
        self._cols = cols
        self._missing_indicator = None
        if len(cols) is 0:
            return base.CallResult(None)

        numerical_inputs = self._inputs.iloc[:,cols]

        missing_indicator = MissingIndicator()
        missing_indicator.fit(numerical_inputs)
        self._missing_indicator = missing_indicator

        return base.CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        if len(self._cols) == 0:
            return base.CallResult(inputs)

        numerical_inputs = inputs.iloc[:,self._cols]
        result = self._missing_indicator.transform(numerical_inputs)

        outputs = inputs.copy()
        for i in range(result.shape[1]):
            outputs[(f'__missing_{i}')] = result[:,i]

        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return
