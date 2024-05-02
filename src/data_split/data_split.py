import argparse
import numpy.random as rnd
from loguru import logger
import os
import tempfile
import yaml
import pandas as pd
import wandb


def time_split_dataset(
    dataset: pd.DataFrame,
    train_start_date: str,
    train_end_date: str,
    holdout_end_date: str,
    time_column: str,
    space_column: str,
    holdout_start_date: str = None,
    split_seed: int = 42,
    space_holdout_percentage: float = 0.1,
) -> tuple:
    """
    Splits a dataset into training, validation, and testing sets based on time and space dimensions.

    The function first segregates the data into time-based training and testing intervals.
    Within the training interval, it further splits the data into training and validation sets
    based on a specified percentage of unique values in the 'space_column', ensuring that
    the validation set is spatially distinct from the training set.

    Parameters:
    - dataset (pd.DataFrame): The complete dataset to split.
    - train_start_date (str): The start date for the training dataset.
    - train_end_date (str): The end date for the training dataset.
    - holdout_end_date (str): The end date for the testing dataset.
    - time_column (str): The column in the dataset that contains the time information.
    - space_column (str): The column in the dataset that represents the spatial information.
    - holdout_start_date (str, optional): The start date for the testing dataset. Defaults to train_end_date.
    - split_seed (int, optional): The seed for the random state used in spatial sampling. Defaults to 42.
    - space_holdout_percentage (float, optional): The percentage of the space_column's unique values
      to hold out for validation.

    Returns:
    - tuple: A tuple containing three pd.DataFrame objects:
        1. train_set: The training dataset.
        2. validation_set: The spatially distinct validation dataset.
        3. test_set: The testing dataset set aside by time.
    """

    state = rnd.RandomState(split_seed)
    holdout_start_date = holdout_start_date if holdout_start_date else train_end_date
    train_set = dataset[
        (dataset[time_column] >= train_start_date)
        & (dataset[time_column] < train_end_date)
    ]
    train_period_space = train_set[space_column].unique()
    test_set = dataset[
        (dataset[time_column] >= holdout_start_date)
        & (dataset[time_column] < holdout_end_date)
    ]
    validation_idx = state.choice(
        a=train_period_space,
        size=int(space_holdout_percentage * len(train_period_space)),
        replace=False,
    )
    validation_set = train_set[train_set[space_column].isin(validation_idx)]
    train_set = train_set[~train_set[space_column].isin(validation_idx)]

    return train_set, validation_set, test_set


def go(args):

    run = wandb.init(job_type="data_pre_process")

    logger.info("Creating artifact")

    logger.info(args.train_data)
    train_data_path = run.use_artifact(args.train_data).file()
    df = pd.read_pickle(train_data_path)

    logger.info("Splitting data into train, val and test")

    with open(args.data_split) as fp:
        model_config = yaml.safe_load(fp)
    wandb.config.update(model_config)

    train_df, validation_df, test_df = time_split_dataset(
        df,
        train_start_date=model_config["train_start_date"],
        train_end_date=model_config["train_end_date"],
        holdout_end_date=model_config["holdout_end_date"],
        time_column=model_config["time_column"],
        space_column=model_config["space_column"],
    )

    splits = {
        "train_data": train_df,
        "validation_data": validation_df,
        "test_data": test_df,
    }

    # Save the artifacts. We use a temporary directory so we do not leave
    # any trace behind
    with tempfile.TemporaryDirectory() as tmp_dir:

        for split, df in splits.items():
            logger.info(split)
            # Make the artifact name from the provided root plus the name of the split
            artifact_name = f"{split}.pickle"

            # Get the path on disk within the temp directory
            temp_path = os.path.join(tmp_dir, artifact_name)

            logger.info(f"Uploading the {split} dataset to {artifact_name}")

            # Save then upload to W&B
            df.to_pickle(temp_path)

            artifact = wandb.Artifact(
                name=artifact_name,
                type=args.artifact_type,
                description=f"{split} split of dataset",
            )
            artifact.add_file(temp_path)

            logger.info("Logging artifact")
            run.log_artifact(artifact)

            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        help="Root for the names of the produced artifacts. The script will produce 2 artifacts: "
        "{root}_train.csv and {root}_test.csv",
        required=True,
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type for the produced artifacts",
        required=True,
    )

    parser.add_argument(
        "--data_split",
        help="Fraction of dataset or number of items to include in the test split",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    go(args)
