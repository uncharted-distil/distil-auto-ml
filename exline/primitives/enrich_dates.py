import os
import sys

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import numpy as np
import pandas as pd

__all__ = ('EnrichDatesPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    pass

class EnrichDatesPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFrame, Hyperparams]):
    """
    A primitive that enriches dates.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'b1367f5b-bab1-4dfc-a1a9-6a56430e516a',
            'version': '0.1.0',
            'name': "Enrich dates",
            'python_path': 'd3m.primitives.data_transformation.data_cleaning.ExlineEnrichDates',
            'source': {
                'name': 'exline',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/cdbethune/d3m-exline/primitives/enrich_dates.py',
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
        print('>> ENRICH_DATES START')
        outputs = inputs.copy()
        outputs = self._enrich_dates(outputs)
        print(outputs)
        print(outputs.dtypes)
        print('<< ENRICH_DATES END')
        return base.CallResult(outputs)

    @classmethod
    def _detect_date(cls, X: container.DataFrame, n_sample: int = 1000) -> bool:
        try:
            # raise Exception # !! Why was this here?
            _ = pd.to_datetime(X.sample(n_sample, replace=True)) # Could also just look at schema
            return True
        except:
            return False

    @classmethod
    def _enrich_dates(cls, inputs: container.DataFrame) -> container.DataFrame:
        cols = list(inputs.columns)
        for c in cols:
            if (inputs[c].dtype == np.object_) and cls._detect_date(inputs[c]):
                print('-- detected date: %s' % c, file=sys.stderr)

                # try:
                inputs_seconds = (pd.to_datetime(inputs[c]) - pd.to_datetime('2000-01-01')).dt.total_seconds().values

                sec_mean = inputs_seconds.mean()
                sec_std  = inputs_seconds.std()

                inputs['%s__seconds' % c] = (inputs_seconds - sec_mean) / sec_std
        return inputs