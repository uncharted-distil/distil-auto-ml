import os
from typing import Set, List, Dict, Any, Optional
from copy import deepcopy

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from modeling.base import EXLineBaseModel
from modeling.metrics import metrics, classification_metrics, regression_metrics
from modeling.helpers import tiebreaking_vote, adjust_f1_macro
from utils import parmap, maybe_subset

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
    metric = hyperparams.Constant[str](
        default='',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    target_idx = hyperparams.Constant[int](
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

class Params(params.Params):
    pass

class EnsembleForestPrimitive(SupervisedLearnerPrimitiveBase[container.ndarray, container.ndarray, Params, Hyperparams]):
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

    default_param_grids = {
        "classification" : {
            "estimator"        : ["RandomForest"],
            "n_estimators"     : [32, 64, 128, 256, 512, 1024, 2048],
            "min_samples_leaf" : [1, 2, 4, 8, 16, 32],
            "class_weight"     : [None, "balanced"],
        },
        "regression" : {
            "estimator"        : ["ExtraTrees", "RandomForest"],
            "bootstrap"        : [True],
            "n_estimators"     : [32, 64, 128, 256, 512, 1024, 2048],
            "min_samples_leaf" : [2, 4, 8, 16, 32, 64],
        }
    }

    def __init__(self,target_metric: Set[str], subset: int = 100000, final_subset: int = 1500000,
        verbose: int = 10, num_fits: int = 1, inner_jobs: int = 1,
        param_grid: Optional[Dict[str, Any]]=None) -> None:

        self.target_metric = target_metric

        if target_metric in classification_metrics:
            self.mode = 'classification'
        elif target_metric in regression_metrics:
            self.mode = 'regression'
        else:
            raise Exception('ForestCV: unknown metric')

        self.subset       = subset
        self.final_subset = final_subset
        self.verbose      = verbose
        self.num_fits     = num_fits
        self.inner_jobs   = inner_jobs
        self.outer_jobs   = 64

        if param_grid is not None:
            self.param_grid = param_grid
        else:
            self.param_grid = deepcopy(self.default_param_grids[self.mode])

        self._models: List[AnyForest]  = []
        self._y_train: Optional[np.ndarray] = None

        self._target_idx = self.hyperparams['target_idx']

    def set_training_data(self, *, inputs: container.ndarray, outputs: container.ndarray) -> None:
        self._inputs = inputs
        self._outputs = outputs[:, self._target_idx]

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        self._models  = [self._fit(self._inputs, self._outputs) for _ in range(self.num_fits)]
        return CallResult(None)

    def produce(self, *, inputs: container.ndarray, timeout: float = None, iterations: int = None) -> CallResult[container.ndarray]:
        preds = [model.predict(inputs) for model in self._models]

        if self.mode == 'classification':
            result = tiebreaking_vote(np.vstack(preds), self._y_train)
        elif self.mode == 'regression':
            result = np.stack(preds).mean(axis=0)
        else:
            result = preds

        return base.CallResult(container.ndarray(result))


    def _fit(self, Xf_train: container.ndarray, y_train: container.ndarray, param_grid: Optional[Dict[str, Any]]=None) -> AnyForest:
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

    @property
    def details(self) -> Dict[str, Any]:
        return {
            "cv_score"    : self.best_fitness,
            "best_params" : self.best_params,
            "num_fits"    : self.num_fits,
        }
