import os
from typing import Set, List, Dict, Any, Optional
from copy import deepcopy

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from exline.modeling.base import EXLineBaseModel
from exline.modeling.metrics import metrics, classification_metrics, regression_metrics
from exline.modeling.helpers import tiebreaking_vote_pre, adjust_f1_macro
from exline.utils import parmap, maybe_subset

import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import ParameterGrid


__all__ = ('EnsembleForest',)

class AnyForest:
    __possible_model_cls = {
        ("regression",     "ExtraTrees")   : ExtraTreesRegressor,
        ("regression",     "RandomForest") : RandomForestRegressor,
        ("classification", "ExtraTrees")   : ExtraTreesClassifier,
        ("classification", "RandomForest") : RandomForestClassifier,
    }

    def __init__(self, mode: Any, estimator: Any, **kwargs: Any):
        assert (mode, estimator) in self.__possible_model_cls
        self.mode   = mode

        self.params    = kwargs
        self.model_cls = self.__possible_model_cls[(mode, estimator)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AnyForest':
        # if self.mode == 'classification':
        #     assert y.dtype == int
        #     assert y.min() == 0, 'may need to remap_labels'
        #     assert y.max() == len(set(y)) - 1, 'may need to remap_labels'

        self.model = self.model_cls(**self.params).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_oob(self) -> np.ndarray:
        if self.mode == 'regression':
            return self.model.oob_prediction_
        elif self.mode == 'classification':
            score_oob = self.model.oob_decision_function_
            return self.model.classes_[score_oob.argmax(axis=-1)] # could vote better

class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Hyperparameter[str](
        default='',
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

    _default_param_grids = {
        # "classification" : {
        #     "estimator"        : ["RandomForest"],
        #     "n_estimators"     : [32, 64, 128, 256, 512, 1024, 2048],
        #     "min_samples_leaf" : [1, 2, 4, 8, 16, 32],
        #     "class_weight"     : [None, "balanced"],
        # },
        # "regression" : {
        #     "estimator"        : ["ExtraTrees", "RandomForest"],
        #     "bootstrap"        : [True],
        #     "n_estimators"     : [32, 64, 128, 256, 512, 1024, 2048],
        #     "min_samples_leaf" : [2, 4, 8, 16, 32, 64],
        # }
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

        self.target_metric = hyperparams['metric']

        if self.target_metric in classification_metrics:
            self.mode = 'classification'
        elif self.target_metric in regression_metrics:
            self.mode = 'regression'
        else:
            raise Exception('ForestCV: unknown metric ' + self.target_metric)

        # were in constructor - can move to hyperparams as needed
        self.subset       = 100000
        self.final_subset = 1500000
        self.verbose      = 10
        self.num_fits     = 1
        self.inner_jobs   = 1
        self.outer_jobs   = 64

        self.param_grid = deepcopy(self._default_param_grids[self.mode])

        self._models: List[AnyForest]  = []
        #self._y_train: Optional[np.ndarray] = None

        #self._target = self.hyperparams['target']

    def __getstate__(self) -> dict:
        state = PrimitiveBase.__getstate__(self)
        state['models'] = self._models
        state['labels'] = self._labels
        return state

    def __setstate__(self, state: dict) -> None:
        PrimitiveBase.__setstate__(self, state)
        self._models = state['models']
        self._labels = state['labels']

    def set_training_data(self, *, inputs: container.DataFrame, outputs: container.DataFrame) -> None:
        self._inputs = inputs.values
        self._outputs = outputs.values

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        self._labels = pd.unique(self._outputs) # store the labels for tie breaking in produce
        self._models  = [self._fit(self._inputs, self._outputs) for _ in range(self.num_fits)]
        return CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        preds = [model.predict(inputs.values) for model in self._models]

        if self.mode == 'classification':
            result = tiebreaking_vote_pre(np.vstack(preds), self._labels)
        elif self.mode == 'regression':
            result = np.stack(preds).mean(axis=0)
        else:
            result = preds

        result_df = container.DataFrame(result)
        return base.CallResult(result_df)


    def _fit(self, Xf_train: np.ndarray, y_train: np.ndarray, param_grid: Optional[Dict[str, Any]]=None) -> AnyForest:
        global _eval_grid_point

        X, y = maybe_subset(Xf_train, y_train, n=self.subset)

        def _eval_grid_point(params: Any) -> Dict[str, Any]:
            model = AnyForest(
                mode=self.mode,
                oob_score=True,
                n_jobs=self.inner_jobs,
                **params
            )
            model       = model.fit(X, y)
            oob_fitness = metrics[self.target_metric](y, model.predict_oob())
            return {"params" : params, "fitness" : oob_fitness}

        # Run grid search
        self.results = parmap(_eval_grid_point,
            ParameterGrid(self.param_grid), verbose=self.verbose, n_jobs=self.outer_jobs)

        # Find best run
        best_run = sorted(self.results, key=lambda x: x['fitness'])[-1] # bigger is better
        self.best_params, self.best_fitness = best_run['params'], best_run['fitness']

        # Refit best model, possibly on more data
        X, y  = maybe_subset(Xf_train, y_train, n=self.final_subset)
        model = AnyForest(mode=self.mode, n_jobs=self.outer_jobs, **self.best_params)
        model = model.fit(X, y)

        return model

    def get_params(self) -> Params:
        return None

    def set_params(self, *, params: Params) -> None:
        return

    @property
    def _details(self) -> Dict[str, Any]:
        return {
            "cv_score"    : self.best_fitness,
            "best_params" : self.best_params,
            "num_fits"    : self.num_fits,
        }
