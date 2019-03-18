import dill

import main_utils as utils

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import (train_test_split,
                                     KFold, StratifiedKFold)

from d3m.container import dataset
from d3m.metadata.pipeline import Pipeline

from exline.external import D3MDataset

from typing import List, Dict, Any, Tuple, Set


from d3m import utils as dutils
PipelineContext = dutils.Enum(value='PipelineContext', names=['TESTING'], start=1)


class Scorer:
    def __init__(self, logger, task, score_config, dats):
        self.logger = logger
        self.solution_id = task.solution_id

        # Assign configurations
        self.dataset_uri = task.dataset_uri
        self.method = score_config.method
        self.metric = score_config.metric
        self.shuffle = score_config.shuffle
        self.random_seed = score_config.random_seed
        self.stratified = score_config.stratified
        self.num_folds = score_config.num_folds
        self.train_size = score_config.train_size
        self.dats = dats

    def run(self):

        # Attempt to load extant 'fit' solution
        #fit_fn = utils.make_job_fn(self.solution_id)
        #with open(fit_fn, 'rb') as f:
        #    unpacked = dill.load(f)
        #    runtime = unpacked['runtime']
        #    fitted_pipeline = unpacked['pipeline']
        runtime = self.dats['runtime']
        fitted_pipeline = self.dats['pipeline']

        # Load the data to be testing with 
        test_dataset = dataset.Dataset.load(self.dataset_uri)
        inputs = [test_dataset]
        #self.logger.info('the vals')
        #self.logger.info(list(test_dataset.keys()).pop())
        #self.logger.info(test_dataset['learningData'].columns)
        #o = runtime.produce(fitted_pipeline, inputs)
        #self.logger.info(o)

        # TODO: actually accept new data
        if self.method == 'holdout':
            return self.hold_out_score()
        elif self.method == 'k_fold':
            return self.k_fold_score()
        else:
            raise ValueError('Cannot score {} type'.format(self.method))

    def _get_features_and_targets(self):
        features = self.engine.variables['features']
        targets = self.engine.variables['targets']
        return features, targets

    def _get_target_name(self):
        cleaned_target_names = False
        target_df = self.engine.variables.get('targets', False)
        if isinstance(target_df, pd.DataFrame):
            target_names = target_df.columns
            cleaned_target_names = [name for name in target_names if name != 'd3m_index']
        return cleaned_target_names

    def _get_pos_label(self):
        """Return pos_label if needed, False if not needed.

        sklearn binary scoring funcs run on indicator types just fine
        but will break on categorical w/o setting pos_label kwarg
        """
        # can safely assume there is only one target for now, will have to change in the future
        labels_series = self.engine.variables['targets'][self._get_target_name()[0]]
        labels_dtype = labels_series.dtype.name
        # not ideal to compare by string name, but direct comparison of the dtype will throw errors
        # for categorical for older versions of pandas
        if labels_dtype == 'object' or labels_dtype == 'category':
            # grab first label arbitrarily bc as of now, no good way to determine what is positive label
            # (not in problem schema or data schema)
            pos_label = labels_series.unique().tolist()[0]
            return pos_label
        return False

    def _binarize(self, true, preds, pos_label):
        lb = LabelBinarizer()
        binary_true = lb.fit_transform(true)
        binary_preds = lb.transform(preds)

        # make sure labels are aligned correctly
        if pos_label and lb.classes_[0] == pos_label:
            return 1 - binary_true, 1 - binary_preds
        return binary_true, binary_preds

    def _f1(self, true, preds):
        pos_label = self._get_pos_label()
        if pos_label:
            return metrics.f1_score(true, preds, pos_label=pos_label)
        return metrics.f1_score(true, preds)

    def _precision(self, true, preds):
        pos_label = self._get_pos_label()
        if pos_label:
            return metrics.precision_score(true, preds, pos_label=pos_label)
        return metrics.precision_score(true, preds)

    def _recall(self, true, preds):
        pos_label = self._get_pos_label()
        if pos_label:
            return metrics.recall_score(true, preds, pos_label=pos_label)
        return metrics.recall_score(true, preds)

    def _roc_score(self, true, preds, average=None):
        pos_label = self._get_pos_label()
        binary_true, binary_preds = self._binarize(true, preds, pos_label)
        if average is not None:
            return metrics.roc_auc_score(binary_true, binary_preds, average=average)
        return metrics.roc_auc_score(binary_true, binary_preds)

    def _rmse_avg(self, true, preds):
        return np.average(metrics.mean_squared_error(true, preds, multioutput='raw_values') ** 0.5)

    def _score(self, metric, true, preds):
        if metric == 'f1_micro':
            score = metrics.f1_score(true, preds, average='micro')
        elif metric == 'f1_macro':
            score = metrics.f1_score(true, preds, average='macro')
        elif metric == 'f1':
            score = self._f1(true, preds)
        elif metric == 'roc_auc':
            score = self._roc_score(true, preds)
        elif metric == 'roc_auc_micro':
            score = self._roc_score(true, preds, average='micro')
        elif metric == 'roc_auc_macro':
            score = self._roc_score(true, preds, average='macro')
        elif metric == 'accuracy':
            score = metrics.accuracy_score(true, preds)
        elif metric == 'precision':
            score = self._precision(true, preds)
        elif metric == 'recall':
            score = self._recall(true, preds)
        elif metric == 'mean_squared_error':
            score = metrics.mean_squared_error(true, preds)
        elif metric == 'root_mean_squared_error':
            score = metrics.mean_squared_error(true, preds) ** 0.5
        elif metric == 'root_mean_squared_error_avg':
            score = self._rmse_avg(true, preds)
        elif metric == 'mean_absolute_error':
            score = metrics.mean_absolute_error(true, preds)
        elif metric == 'r_squared':
            score = metrics.r2_score(true, preds)
        elif metric == 'jaccard_similarity_score':
            score = metrics.jaccard_similarity_score(true, preds)
        elif metric == 'normalized_mutual_information':
            score = metrics.normalized_mutual_info_score(true, preds)
        else:
            raise ValueError('Cannot score metric {}'.format(metric))
        return score

    def hold_out_score(self):
        
        features, targets = self._get_features_and_targets()
        
        labels = targets.values if self.stratified else None

        X_train, X_test, y_train, y_test = train_test_split(features,
                                                            targets,
                                                            train_size=self.train_size,
                                                            test_size=None,
                                                            shuffle=self.shuffle,
                                                            random_state=self.random_seed,
                                                            stratify=labels)

        self.engine.variables['features'] = X_train
        self.engine.variables['targets'] = y_train
        self.engine.refit()

        result = self.engine.model_produce(X_test)
        score = self._score(self.metric, y_test, result)
        
        return [score]

    def k_fold_score(self,):
        fold_scores = []
        kf = StratifiedKFold if self.stratified else KFold
        kf = kf(n_splits=self.num_folds, shuffle=self.shuffle, random_state=self.random_seed)

        features, targets = self._get_features_and_targets()

        for train_index, test_index in kf.split(features, targets):
            X_train, X_test = features.iloc[train_index, :], features.iloc[test_index, :]
            y_train, y_test = targets.iloc[train_index, :], targets.iloc[test_index, :]

            self.engine.variables['features'] = X_train
            self.engine.variables['targets'] = y_train
            self.engine.refit()

            result = self.engine.model_produce(X_test)
            score = self._score(self.metric, y_test, result)
            fold_scores.append(score)
            
        return fold_scores
