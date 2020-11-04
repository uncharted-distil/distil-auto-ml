import numpy as np
from sklearn import metrics as sklearn_metrics
from d3m.metrics import (
    HitsAtKMetric,
    MeanReciprocalRankMetric,
    RocAucMicroMetric,
    RocAucMacroMetric,
    RocAucMetric,
)

# from external import objectDetectionAP

hits_at_k = HitsAtKMetric(5)  # todo how does this get set?
mean_recriprocal_rank = MeanReciprocalRankMetric()
roc_auc_micro = RocAucMicroMetric()
roc_auc_macro = RocAucMacroMetric()
roc_auc = RocAucMetric()


metrics = {
    # classification
    "f1Macro": lambda act, pred: sklearn_metrics.f1_score(act, pred, average="macro"),
    "f1Micro": lambda act, pred: sklearn_metrics.f1_score(act, pred, average="micro"),
    "f1": lambda act, pred: sklearn_metrics.f1_score(act, pred),
    "accuracy": lambda act, pred: sklearn_metrics.accuracy_score(act, pred),
    # regression
    "meanSquaredError": lambda act, pred: -1.0
    * sklearn_metrics.mean_squared_error(act, pred),
    "meanAbsoluteError": lambda act, pred: -1.0
    * sklearn_metrics.mean_absolute_error(act, pred),
    "rootMeanSquaredError": lambda act, pred: -1.0
    * np.sqrt(sklearn_metrics.mean_squared_error(act, pred)),
    "rootMeanSquaredErrorAvg": lambda act, pred: -1.0
    * np.sqrt(sklearn_metrics.mean_squared_error(act, pred)),
    "rSquared": lambda act, pred: -1.0 * sklearn_metrics.r2_score(act, pred),
    # clustering
    "normalizedMutualInformation": sklearn_metrics.normalized_mutual_info_score,
    # object detection
    "objectDetectionAP": lambda act, pred: 0.0,
    "meanReciprocalRank": lambda act, pred: mean_recriprocal_rank.score(act, pred),
    "hitsAtK": lambda act, pred: hits_at_k.score(act, pred),
    "rocAucMacro": lambda act, pred: roc_auc_macro.score(act, pred),
    "rocAucMicro": lambda act, pred: roc_auc_micro.score(act, pred),
    "rocAuc": lambda act, pred: roc_auc.score(act, pred),
}

classification_metrics = set(
    [
        "f1Macro",
        "f1Micro",
        "f1",  # binary
        "accuracy",
        "rocAuc",  # weighted
        "rocAucMicro",
        "rocAucMacro",
    ]
)

confidence_metrics = set(["rocAuc", "rocAucMacro", "rocAucMicro"])

binary_classification_metrics = set(
    [
        "f1Macro",
        "f1Micro",
        "f1",  # binary
        "accuracy",
        "rocAuc",  # weighted
        "rocAucMacro",
        "rocAucMicro",
    ]
)

multiclass_classification_metrics = set(
    [
        "f1Macro",
        "f1Micro",
        "accuracy",
        "rocAuc",  # weighted
        "rocAucMacro",
    ]
)

regression_metrics = set(
    [
        "meanSquaredError",
        "meanAbsoluteError",
        "rootMeanSquaredError",
        "rootMeanSquaredErrorAvg",
        "rSquared",
    ]
)

clustering_metrics = set(
    [
        "normalizedMutualInformation",
    ]
)

d3m_lookup = {
    "f1Macro": "f1_macro",
    "f1Micro": "f1_micro",
    "f1": "f1",
    "accuracy": "accuracy",
    "rSquared": "r_squared",
    "meanSquaredError": "mean_squared_error",
    "rootMeanSquaredError": "root_mean_squared_error",
    "rootMeanSquaredErrorAvg": "root_mean_squared_error_avg",
    "meanAbsoluteError": "mean_absolute_error",
    "normalizedMutualInformation": "normalized_mutual_information",
    "objectDetectionAP": "object_detection_average_precision",
    "meanReciprocalRank": "mean_reciprocal_rank",
    "hitsAtK": "hits_at_k",
    "rocAucMacro": "roc_auc_macro",
    "rocAucMicro": "roc_auc_micro",
    "rocAuc": "roc_auc",
}

inverse_d3m_lookup = {v: k for k, v in d3m_lookup.items()}


def translate_d3m_metric(metric):
    assert metric in d3m_lookup, "%s not in lookup" % metric
    return d3m_lookup[metric]


def translate_metric(d3m_metric):
    assert d3m_metric in inverse_d3m_lookup, "%s not in lookup" % d3m_metric
    return inverse_d3m_lookup[d3m_metric]


def translate_proto_metric(proto_metric):
    lookup = {
        "F1_MACRO": "f1Macro",
        "F1_MICRO": "f1Micro",
        "F1": "f1",
        "ACCURACY": "accuracy",
        "MEAN_SQUARED_ERROR": "meanSquaredError",
        "ROOT_MEAN_SQUARED_ERROR": "rootMeanSquaredError",
        "ROOT_MEAN_SQUARED_ERROR_AVG": "rootMeanSquaredErrorAvg",
        "R_SQUARED": "rSquared",  # mapped for now,
        "MEAN_ABSOLUTE_ERROR": "meanAbsoluteError",
        "NORMALIZED_MUTUAL_INFORMATION": "normalizedMutualInformation",
        "OBJECT_DETECTION_AVERAGE_PRECISION": "objectDetectionAP",
        "MEAN_RECIPROCAL_RANK": "meanReciprocalRank",  # todo add this to primitives metrics
        "HITS_AT_K": "hitsAtK",
        "ROC_AUC_MACRO": "rocAucMacro",
        "ROC_AUC_MICRO": "rocAucMicro",
        "ROC_AUC": "rocAuc",
    }
    assert proto_metric in lookup, "%s not in lookup" % proto_metric
    return lookup[proto_metric]
