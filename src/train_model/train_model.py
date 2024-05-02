import os
from loguru import logger
import wandb
import mlflow
from sklearn import metrics
from scipy.stats import ks_2samp
import argparse
import yaml
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from mlflow.models import infer_signature
import tempfile


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


def go(args):
    run = wandb.init(job_type="train")

    logger.info("Downloading and reading train artifact")

    train_data_path = run.use_artifact(args.train_data).file()
    train_df = pd.read_pickle(train_data_path)

    validation_data_path = run.use_artifact(args.validation_data).file()
    validation_df = pd.read_pickle(validation_data_path)

    with open(args.model_config) as fp:
        model_config = yaml.safe_load(fp)
    wandb.config.update(model_config)

    params = model_config["xgbm"]
    split_seed = model_config["random_seed"]
    features = model_config["features"]["numerical"]
    target = model_config["target"]

    xgbm_model = XGBClassifier(missing=np.nan, **params, random_state=split_seed)

    X_train, Y_train = train_df[features], train_df[target]
    X_valid, Y_valid = validation_df[features], validation_df[target]

    xgbm_model.fit(X_train, Y_train)
    xgbm_preds = xgbm_model.predict_proba(X_valid)[:, 1]

    model_results = calculate_metrics(Y_valid, xgbm_preds)

    run.summary["ROC_AUC"] = model_results["roc_auc_score"]
    run.summary["PR_AUC"] = model_results["pr_auc"]
    run.summary["KS"] = model_results["ks"]

    ### Export if required
    if args.export_artifact != "null":
        export_model(
            run, xgbm_model, features, X_valid, xgbm_preds, args.export_artifact
        )

    run.log({"model_results": model_results})


def export_model(run, pipe, used_columns, X_val, val_pred, export_artifact):

    # Infer the signature of the model

    # Get the columns that we are really using from the pipeline
    signature = infer_signature(X_val[used_columns], val_pred)

    with tempfile.TemporaryDirectory() as temp_dir:

        export_path = os.path.join(temp_dir, "model_export")

        mlflow.sklearn.save_model(
            pipe,
            export_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val.iloc[:2],
        )

        artifact = wandb.Artifact(
            export_artifact,
            type="model_export",
            description="Random Forest pipeline export",
        )
        artifact.add_dir(export_path)

        run.log_artifact(artifact)

        # Make sure the artifact is uploaded before the temp dir
        # gets deleted
        artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest",
        fromfile_prefix_chars="@",
    )

    logger.info("--train_data")
    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    logger.info("--validation_data")
    parser.add_argument(
        "--validation_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    logger.info("--model_config")
    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a JSON file containing the configuration for the random forest",
        required=True,
    )

    logger.info("--export_artifact")
    parser.add_argument(
        "--export_artifact",
        type=str,
        help="Name of the artifact for the exported model. Use 'null' for no export.",
        required=False,
        default="null",
    )

    args = parser.parse_args()

    go(args)
