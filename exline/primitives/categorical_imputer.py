import os
from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import pandas as pd
import numpy as np

from sklearn_pandas import CategoricalImputer
from sklearn import compose

from preprocessing.utils import MISSING_VALUE_INDICATOR


__all__ = ('CategoricalImputerPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )

class CategoricalImputerPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFrame, Hyperparams]):
    """
    A primitive that imputes categoricals.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '0a9936f3-7784-4697-82f0-2a5fcc744c16',
            'version': '0.1.0',
            'name': "Categorical imputer",
            'python_path': 'd3m.primitives.data_transformation.imputer.ExlineCategoricalImputer',
            'source': {
                'name': 'exline',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/cdbethune/d3m-exline/primitives/categorical_imputer.py',
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
        print('>> CATEGORICAL IMPUTER START')
        cols = self.hyperparams['use_columns']
        input_cols = inputs.iloc[:,cols]

        imputer = CategoricalImputer(strategy='constant', fill_value=MISSING_VALUE_INDICATOR)
        imputer.fit(input_cols)
        result = imputer.transform(input_cols)

        outputs = inputs.copy()
        for idx, col_idx in enumerate(cols):
            outputs.iloc[:,col_idx] = result[:,idx]

        print(outputs)
        print(outputs.dtypes)
        print('<< CATEGORICAL IMPUTER END')
        return base.CallResult(outputs)
