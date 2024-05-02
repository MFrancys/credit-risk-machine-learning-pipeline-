import argparse
from pathlib import Path
from loguru import logger
from datetime import datetime
import pandas as pd
import wandb

from credit_report_features.make_features import build_credit_report_features
from previous_internal_application_features.make_features import build_previous_internal_app_features


def go(args):
    run = wandb.init(job_type="build_data")

    logger.info("Creating artifact...")

    logger.info("Build previous_internal_apps_features...")
    df = pd.read_parquet(args.main_dataset_path_artifact)
    df["loan_origination_datetime_month"] = df["LOAN_ORIGINATION_DATETIME"].dt.strftime("%Y-%m")


    credit_report_df = pd.read_parquet(args.credit_report_dataset_path_artifact)
    credit_report_df = pd.merge(credit_report_df,
                                df[["LOAN_ORIGINATION_DATETIME", "customer_id", "APPLICATION_DATETIME"]], how="left",
                                on="customer_id")
    df_previous_internal_apps_features = build_previous_internal_app_features(df)

    logger.info("Build credit_reports_features...")
    df_credit_reports_features = build_credit_report_features(credit_report_df)
    df_credit_reports_features["credit_reports__loans_count"] = df_credit_reports_features[
        "credit_reports__loans_count"].fillna(0)

    logger.info("Merge Data...")
    final_df = pd.merge(
        df[["customer_id", "loan_id", "target", "loan_origination_datetime_month"]],
        df_previous_internal_apps_features,
        how="left",
        on="loan_id"
    )

    final_df = pd.merge(
        final_df,
        df_credit_reports_features,
        how="left",
        on="customer_id"
    )
    final_df = final_df.fillna(0)

    formatted_date = datetime.now().strftime("%Y%m%d")
    store_data_path = Path.cwd().parent.parent / f"models/{formatted_date}_final_dataset.pickle"
    logger.info(store_data_path)

    final_df.to_pickle(store_data_path)

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(Path.joinpath(store_data_path))
    
    logger.info("Logging artifact")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--main_dataset_path_artifact",
        type=str,
        help="Path for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--credit_report_dataset_path_artifact",
        type=str,
        help="Path for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)