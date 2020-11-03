import copy

import numpy as np
import pandas as pd
from d3m.container import dataset
from processing import pipeline
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer


class Scorer:

    D3M_INDEX_IDX = 0
    PREDICTION_IDX = 1
    CONFIDENCE_IDX = 2

    def __init__(self, logger, task, score_config, fitted_pipeline, target_idx):
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
        self.fitted_pipeline = fitted_pipeline
        self.target_idx = target_idx

    def run(self):

        # Attempt to load extant 'fit' solution
        # fit_fn = utils.make_job_fn(self.solution_id)
        # with open(fit_fn, 'rb') as f:
        #    unpacked = dill.load(f)
        #    runtime = unpacked['runtime']
        #    fitted_pipeline = unpacked['pipeline']

        # Load the data to test
        self.inputs = dataset.Dataset.load(self.dataset_uri)

        # TODO: actually accept new data
        if self.method == "holdout":
            return self.hold_out_score()
        elif self.method == "ranking":
            return self.ranking()
        # elif self.method == 'k_fold':
        #    #return self.k_fold_score()
        else:
            raise ValueError("Cannot score {} type".format(self.method))

    def _get_pos_label(self, labels_series):
        """Return pos_label if needed, False if not needed.

        sklearn binary scoring funcs run on indicator types just fine
        but will break on categorical w/o setting pos_label kwarg
        """
        # can safely assume there is only one target for now, will have to change in the future
        labels_dtype = labels_series.dtype.name
        # not ideal to compare by string name, but direct comparison of the dtype will throw errors
        # for categorical for older versions of pandas
        if labels_dtype == "object" or labels_dtype == "category":
            # (not in problem schema or data schema)
            labels_list = labels_series.unique().tolist()
            # since problem / data schema don't list positive label, we'll do a quick heuristic
            if set(labels_list) == set(["0", "1"]):
                return "1"
            else:
                # grab first label arbitrarily bc as of now, no good way to determine what is positive label
                return labels_list[0]
        return False

    def _binarize(self, true, labels):
        lb = LabelBinarizer()
        binary_true = lb.fit_transform(true)
        binary_preds = lb.transform(preds)

        # make sure labels are aligned correctly
        if pos_label and lb.classes_[0] == pos_label:
            return 1 - binary_true, 1 - binary_preds
        return binary_true, binary_preds

    def _f1(self, true, preds):
        pos_label = self._get_pos_label(true)
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

    def _roc_score(self, true, confidences, average=None):
        # need to binarize labels in the multi
        if average is not None:
            return metrics.roc_auc_score(
                true, confidences, average=average, multi_class="ovr"
            )
        return metrics.roc_auc_score(true, confidences)

    def _rmse_avg(self, true, preds):
        return np.average(
            metrics.mean_squared_error(true, preds, multioutput="raw_values") ** 0.5
        )

    def _score(self, metric, true, preds, confidences=None):
        if metric == "f1_micro":
            score = metrics.f1_score(true, preds, average="micro")
        elif metric == "f1_macro":
            score = metrics.f1_score(true, preds, average="macro")
        elif metric == "f1":
            score = self._f1(true, preds)
        elif metric == "roc_auc":
            score = self._roc_score(true, confidences)
        elif metric == "roc_auc_micro":
            score = self._roc_score(true, confidences, average="micro")
        elif metric == "roc_auc_macro":
            score = self._roc_score(true, confidences, average="macro")
        elif metric == "accuracy":
            score = metrics.accuracy_score(true, preds)
        elif metric == "precision":
            score = self._precision(true, preds)
        elif metric == "recall":
            score = self._recall(true, preds)
        elif metric == "mean_squared_error":
            score = metrics.mean_squared_error(true, preds)
        elif metric == "root_mean_squared_error":
            score = metrics.mean_squared_error(true, preds) ** 0.5
        elif metric == "root_mean_squared_error_avg":
            score = self._rmse_avg(true, preds)
        elif metric == "mean_absolute_error":
            score = metrics.mean_absolute_error(true, preds)
        elif metric == "r_squared":
            score = metrics.r2_score(true, preds)
        elif metric == "jaccard_similarity_score":
            score = metrics.jaccard_similarity_score(true, preds)
        elif metric == "normalized_mutual_information":
            score = metrics.normalized_mutual_info_score(true, preds)
        elif metric == "object_detection_average_precision":
            self.logger.warning(f"{metric} metric unsuppported - returning 0")
            score = 0.0
        else:
            raise ValueError("Cannot score metric {}".format(metric))
        return score

    def hold_out_score(self):
        # produce predictions from the fitted model and extract to single col dataframe
        # with the d3mIndex as the index
        _in = copy.deepcopy(self.inputs)
        results = pipeline.produce(self.fitted_pipeline, _in)
        # Not sure how to do this properly - we assume that we will use `outputs.0` for scoring, but it is
        # possible that, in a non-standard pipeline, `outputs.0` could be the output from another step,
        # and `outputs.1` contains the predictions.
        if len(results.values) > 1:
            self.logger.warning(
                "Pipeline produced > 1 outputs. Scoring first output only."
            )
        result_df = results.values["outputs.0"]

        # get column names for convenience
        d3m_index_col = result_df.columns[Scorer.D3M_INDEX_IDX]
        confidence_col = (
            result_df.columns[Scorer.CONFIDENCE_IDX]
            if len(result_df.columns) > 2
            else None
        )
        prediction_col = result_df.columns[Scorer.PREDICTION_IDX]

        # when a confidence column is present, we need to make sure the data is formatted such that it
        # can be passed to the sklearn scoring functions
        confidence_matrix = None
        if confidence_col:
            # Results returned by pipelines are formatted as [d3mIndex, prediction, confidence (optional)]
            # where results with confidences for binary problems have unique d3mIndex values for each row.
            # Muliclass problems with confidences use a multi-index approach, where the labels of each
            # prediction are assigned the same d3mIndex.
            if not result_df[d3m_index_col].is_unique:
                # Convert into a n_classes x n_sample matrix, where we take each row that shares a d3mIndex
                # convert them into rows of the matrix.  This is the required format for downstream
                # scoring functions.
                confidence_matrix = np.stack(
                    result_df.groupby(d3m_index_col)[confidence_col]
                    .apply(np.array)
                    .values
                )

            # Get the predictions into sorted order by d3mIndex, confidence.
            result_df[d3m_index_col] = pd.to_numeric(result_df[d3m_index_col])
            result_df[confidence_col] = pd.to_numeric(result_df[confidence_col])
            result_df.sort_values(
                by=[d3m_index_col, confidence_col],
                ascending=[True, False],
                inplace=True,
            )
        else:
            # no confidences, just ensure that the result is sorted by D3M index to ensure consistency
            # with the ground truth.
            result_df.sort_values(by=[d3m_index_col], inplace=True)

        # take one label in case this is a multi index - previous sort should guarantee
        # the top label is taken if confidences were assigned.  This is the required format
        # for metrics that just score on the label.
        result_df.drop_duplicates(inplace=True, subset=d3m_index_col)

        # put the ground truth into a single col dataframe
        true_df = self.inputs["learningData"]
        true_df[d3m_index_col] = pd.to_numeric(true_df[d3m_index_col])

        # take one label in the case this is a multi index
        true_df.drop_duplicates(inplace=True, subset=d3m_index_col)
        true_df.sort_values(by=[d3m_index_col], inplace=True)

        # only take the d3m indices that exist for results
        true_df = true_df.set_index(true_df[d3m_index_col])
        result_df = result_df.set_index(result_df[d3m_index_col])
        true_df = true_df.loc[result_df.index]

        result_series = result_df[prediction_col]
        result_series.fillna(value=0, inplace=True)
        true_series = true_df.iloc[:, self.target_idx]

        # force the truth value to the same type as the predicted value
        true_series = true_series.astype(result_series.dtype)

        # if we have a confidence matrix, use that to score, otherwise use the the contents
        # of the confidence column if defined
        confidence = None
        if confidence_col:
            if confidence_matrix is not None:
                confidence = confidence_matrix
            else:
                confidence = pd.to_numeric(result_df.iloc[:, confidence_col])

        return [self._score(self.metric, true_series, result_series, confidence)]

    def ranking(self):
        # rank is always 1 when requested since the system only generates a single solution
        if self.metric == "rank":
            score = [1]
        else:
            raise ValueError(f"Cannot rank metric {self.metric}")
        return score

    """
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
    """
