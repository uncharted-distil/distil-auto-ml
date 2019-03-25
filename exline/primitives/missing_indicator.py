import os
import logging

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

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


class MissingIndicatorPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFrame, Hyperparams]):
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
        logger.debug(f'Running {__name__}')

        cols = list(self.hyperparams['use_columns'])

        if cols is None or len(cols) is 0:
            cols = []
            for idx, c in enumerate(inputs.columns):
                if inputs[c].dtype == int or inputs[c].dtype == float or inputs[c].dtype == bool:
                    cols.append(idx)

        logger.debug(f'Found {len(cols)} cols to check for missing values')

        if len(cols) is 0:
            return base.CallResult(inputs)

        numerical_inputs = inputs.iloc[:,cols]

        missing_indicator = MissingIndicator()
        missing_indicator.fit(numerical_inputs)
        result = missing_indicator.transform(numerical_inputs)

        outputs = inputs.copy()
        for i, c in enumerate(cols):
            outputs.iloc[:, c] = result[:, i]

        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)