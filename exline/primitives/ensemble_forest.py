import os
import logging
from typing import Set, List, Dict, Any, Optional

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from exline.modeling.forest import ForestCV
from exline.modeling.metrics import classification_metrics, regression_metrics

import pandas as pd
import numpy as np

__all__ = ('EnsembleForest',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Hyperparameter[str](
        default='',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    fast = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class Params(params.Params):
    pass

class EnsembleForestPrimitive(PrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    A primitive that forests.
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '8baea8e6-9d3a-46d7-acf1-04fd593dcd37',
            'version': '0.1.0',
            'name': "EnsembleForest",
            'python_path': 'd3m.primitives.learner.random_forest.ExlineEnsembleForest',
            'source': {
                'name': 'exline',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/cdbethune/d3m-exline/primitives/ensemble_forest.py',
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

    # number of rows to limit to when in fast mode
    _FAST_FIT_ROWS = 500

    # grids to use when in fast mode
    _FAST_GRIDS = {
        "classification" : {
            "estimator"        : ["RandomForest"],
            "n_estimators"     : [32],
            "min_samples_leaf" : [1],
            "class_weight"     : [None],
        },
        "regression" : {
            "estimator"        : ["ExtraTrees", "RandomForest"],
            "bootstrap"        : [True],
            "n_estimators"     : [32],
            "min_samples_leaf" : [2],
        }
    }

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:

        PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed)

        self._grid = self._get_grid_for_metric() if self.hyperparams['fast'] else None
        self._model = ForestCV(self.hyperparams['metric'], param_grid=self._grid)

    def __getstate__(self) -> dict:
        state = PrimitiveBase.__getstate__(self)
        state['models'] = self._model
        state['grid'] = self._grid
        return state

    def __setstate__(self, state: dict) -> None:
        PrimitiveBase.__setstate__(self, state)
        self._model = state['models']
        self._grid = state['grid']

    def set_training_data(self, *, inputs: container.DataFrame, outputs: container.DataFrame) -> None:
        self._inputs = inputs
        self._outputs = outputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        logger.debug(f'Fitting {__name__}')
        if self.hyperparams['fast']:
            rows = len(self._inputs.index)
            if rows > self._FAST_FIT_ROWS:
                sampled_inputs = self._inputs.sample(n=self._FAST_FIT_ROWS, random_state=1)
                sampled_outputs = self._outputs.loc[self._outputs.index.intersection(sampled_inputs.index), ]
                self._model.fit(sampled_inputs, sampled_outputs)
        else:
            self._model.fit(self._inputs, self._outputs)
        return CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        # create dataframe to hold d3mIndex and result
        result = self._model.predict(inputs)
        result_df = container.DataFrame({inputs.index.name: inputs.index, self._outputs.columns[0]: result}, generate_metadata=True)

        # mark the semantic types on the dataframe
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        logger.debug(f'\n{result_df}')
        return base.CallResult(result_df)

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return

    def _get_grid_for_metric(self) -> Dict[str, Any]:
        if self.hyperparams['metric'] in classification_metrics:
            return self._FAST_GRIDS['classification']
        elif self.hyperparams['metric'] in regression_metrics:
            return self._FAST_GRIDS['regression']
        else:
            raise Exception('ForestCV: unknown metric')