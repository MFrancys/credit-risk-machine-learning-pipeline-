import mlflow
from pathlib import Path
from loguru import logger
import os
import json
import wandb
from pathlib import Path
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name="config")
def go(config: DictConfig):

    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    logger.info("Get the path at the root of the MLflow project")
    root_path = hydra.utils.get_original_cwd()
    logger.info(root_path)

    update_root_path = Path(root_path).parent
    main_dataset_path_artifact = (
        update_root_path / config["main"]["data_path"]["main_dataset"]
    )
    logger.info(main_dataset_path_artifact)

    credit_report_dataset_path_artifact = (
        update_root_path / config["main"]["data_path"]["credit_report_dataset"]
    )
    logger.info(credit_report_dataset_path_artifact)

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:

        steps_to_execute = list(config["main"]["execute_steps"])

    print(config["main"]["data_path"])

    if "build_data" in steps_to_execute:
        logger.info("============Build Dataset============")
        _ = mlflow.run(
            os.path.join(root_path, "build_data"),
            "main",
            parameters={
                "main_dataset_path_artifact": main_dataset_path_artifact,
                "credit_report_dataset_path_artifact": credit_report_dataset_path_artifact,
                "artifact_name": "final_dataset.pickle",
                "artifact_type": "final_dataset",
                "artifact_description": "Train data with preprocessing applied",
            },
        )

    if "data_split" in steps_to_execute:
        logger.info("============Data Split Step============")

        model_config = os.path.abspath("split_data.yml")
        logger.info(model_config)

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["data_split"]))

        _ = mlflow.run(
            os.path.join(root_path, "data_split"),
            "main",
            parameters={
                "train_data": "final_dataset.pickle:latest",
                "artifact_root": "data",
                "artifact_type": "split_data",
                "data_split": model_config,
            },
        )

    if "train_model" in steps_to_execute:
        logger.info("============Train model Step============")
        model_config = os.path.abspath("model_config.yml")
        logger.info(model_config)

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["model_pipeline"]))

        _ = mlflow.run(
            os.path.join(root_path, "train_model"),
            "main",
            parameters={
                "train_data": config["data"]["train_data"],
                "validation_data": config["data"]["validation_data"],
                "model_config": model_config,
                "export_artifact": config["model_pipeline"]["export_artifact"],
            },
        )

    if "evaluate_model" in steps_to_execute:
        logger.info("============Evaluate model Step============")
        model_config = os.path.abspath("model_config.yml")
        logger.info(model_config)

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["model_pipeline"]))

        _ = mlflow.run(
            os.path.join(root_path, "evaluate_model"),
            "main",
            parameters={
                "model_export": f"{config['model_pipeline']['export_artifact']}:latest",
                "test_data": config["data"]["test_data"],
                "model_config": model_config,
            },
        )


if __name__ == "__main__":
    go()
