import numpy as np
from sklearn import metrics as sklearn_metrics

# from external import objectDetectionAP

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
}

classification_metrics = set(["f1Macro", "f1Micro", "f1", "accuracy"])

regression_metrics = set(
    [
        "meanSquaredError",
        "meanAbsoluteError",
        "rootMeanSquaredError",
        "rootMeanSquaredErrorAvg",
        "rSquared",
    ]
)

clustering_metrics = set(["normalizedMutualInformation",])


def translate_d3m_metric(metric):
    lookup = {
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
    }
    assert metric in lookup, "%s not in lookup" % metric
    return lookup[metric]


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
    }
    assert proto_metric in lookup, "%s not in lookup" % proto_metric
    return lookup[proto_metric]
