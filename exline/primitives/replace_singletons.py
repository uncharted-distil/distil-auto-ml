import os

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from .utils import SINGLETON_INDICATOR

import typing
import numpy as np
import pandas as pd

__all__ = ('ReplaceSingletonsPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    keep_text = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='keeps text')


class ReplaceSingletonsPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFarme, Hyperparams]):
    """
    A primitive that replaces singletons.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '7cacc8b6-85ad-4c8f-9f75-360e0faee2b8',
            'version': '0.1.0',
            'name': "Replace singeltons",
            'python_path': 'd3m.primitives.data_transformation.data_cleaning.ExlineReplaceSingletons',
            'source': {
                'name': 'exline',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/cdbethune/d3m-exline/primitives/replace_singletons.py',
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
        """ set values that only occur once to a special token """

        cols: typing.List[container.DataFrame] = list(inputs.columns)
        for c in cols:
            if inputs[c].dtype == np.object_:
                if not self.hyperparams['keep_text'] or not self._detect_text(inputs[c]):
                    vcs = pd.value_counts(list(inputs[c]))
                    singletons = set(vcs[vcs == 1].index)
                    if singletons:
                        inputs[c][inputs[c].isin(singletons)] = SINGLETON_INDICATOR
        return base.CallResult(container.DataFrame(inputs))

    @classmethod
    def _detect_text(cls, X: container.DataFrame, thresh: int = 8) -> bool:
        """ returns true if median entry has more than `thresh` tokens"""
        X = X[X.notnull()]
        n_toks = X.apply(lambda xx: len(str(xx).split(' '))).values
        return np.median(n_toks) >= thresh