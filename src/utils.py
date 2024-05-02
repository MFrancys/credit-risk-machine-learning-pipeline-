import numpy as np
from sklearn import metrics
from scipy.stats import ks_2samp
import pickle


def calculate_metrics(y_test: np.ndarray, preds: np.ndarray) -> dict:
    """
    Calculates several key performance metrics for evaluating a classification model.

    This function computes the following metrics:
    - ROC AUC Score: The area under the Receiver Operating Characteristic curve, useful for assessing the
      overall effectiveness of the predictions with respect to the true outcomes.
    - PR AUC Score: The area under the Precision-Recall curve, useful for datasets with a significant imbalance.
    - Kolmogorov-Smirnov Statistic (KS): Measures the degree to which the distributions of the predicted
      probabilities of the positive and negative classes differ.

    Parameters:
    - y_test (np.ndarray): An array containing the actual binary labels of the test data (0 or 1).
    - preds (np.ndarray): An array containing the predicted probabilities corresponding to the likelihood
      of the positive class (class label 1).

    Returns:
    - dict: A dictionary containing the calculated metrics:
        * 'roc_auc_score': float, representing the ROC AUC score.
        * 'pr_auc': float, representing the precision-recall AUC score.
        * 'ks': float, representing the Kolmogorov-Smirnov statistic.
    """

    metrics_dict = {
        "roc_auc_score": metrics.roc_auc_score(y_true=y_test, y_score=preds),
        "pr_auc": metrics.average_precision_score(y_true=y_test, y_score=preds),
        "ks": ks_2samp(preds[y_test == 0], preds[y_test == 1])[0],
    }

    return metrics_dict


def save_pickle(object: list, path: str) -> None:
    with open(path, "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> object:
    with open(path, "rb") as handle:
        loaded_object = pickle.load(handle)
    return loaded_object
