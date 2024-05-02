from loguru import logger
import wandb
import mlflow
from sklearn import metrics
from scipy.stats import ks_2samp
import argparse
import yaml
import pandas as pd
import numpy as np


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
        'roc_auc_score': metrics.roc_auc_score(y_true=y_test, y_score=preds),
        'pr_auc': metrics.average_precision_score(y_true=y_test, y_score=preds),
        'ks': ks_2samp(preds[y_test == 0], preds[y_test == 1])[0],
    }

    return metrics_dict


def go(args):

    run = wandb.init(job_type="test")

    logger.info("Downloading and reading test artifact")

    with open(args.model_config) as fp:
        model_config = yaml.safe_load(fp)
    wandb.config.update(model_config)
    features = model_config["features"]["numerical"]
    target = model_config["target"]

    test_data_path = run.use_artifact(args.test_data).file()
    test_df = pd.read_csv(test_data_path, low_memory=False)

    logger.info("Extracting target from dataframe")
    X_test, Y_test = test_df[features], test_df[target]

    logger.info("Downloading and reading the exported model")
    model_export_path = run.use_artifact(args.model_export).download()

    pipe = mlflow.sklearn.load_model(model_export_path)
    pred_proba = pipe.predict_proba(X_test[features])[:, 1]

    logger.info("Scoring")

    model_results = calculate_metrics(Y_test, pred_proba)

    run.summary["ROC_AUC"] = model_results['roc_auc_score']
    run.summary["PR_AUC"] = model_results['pr_auc']
    run.summary["KS"] = model_results['ks']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    logger.info("--model_config")
    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a JSON file containing the configuration for the random forest",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    args = parser.parse_args()

    go(args)
