"""
Registers trained ML model if deploy flag is True.
"""

import argparse
import json
import os
from pathlib import Path

import mlflow


def parse_args() -> argparse.Namespace:
    """Parses the input arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Registers the ML model.")
    parser.add_argument(
        "--model_name", type=str, help="Name under which model will be registered."
    )
    parser.add_argument("--model_path", type=str, help="Model directory.")
    parser.add_argument(
        "--evaluation_output", type=str, help="Path of evaluation results."
    )
    parser.add_argument(
        "--model_info_output_path", type=str, help="Path to write model info to JSON."
    )

    args, _ = parser.parse_known_args()
    print(f"Arguments: {args}")

    return args


def main(args: argparse.Namespace) -> None:
    """
    Main function that registers the model.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    with open((Path(args.evaluation_output) / "deploy_flag"), "rb") as f:
        deploy_flag = int(f.read())

    mlflow.log_metric("deploy flag", int(deploy_flag))

    deploy_flag = 1
    if deploy_flag == 1:
        print("Registering ", args.model_name)

        # Load model
        model = mlflow.sklearn.load_model(args.model_path)

        # Log model using mlflow
        mlflow.sklearn.log_model(model, args.model_name)

        # Register logged model using mlflow
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{args.model_name}"
        mlflow_model = mlflow.register_model(model_uri, args.model_name)
        model_version = mlflow_model.version

        # Write model info
        print("Writing model info...")
        dict = {"id": "{0}:{1}".format(args.model_name, model_version)}
        output_path = os.path.join(args.model_info_output_path, "model_info.json")
        with open(output_path, "w") as f:
            json.dump(dict, fp=f)

    else:
        print("Model will not be registered!")


if __name__ == "__main__":

    # Start an MLflow run
    with mlflow.start_run():

        # Parse the input arguments
        args = parse_args()

        lines = [
            f"Model name: {args.model_name}",
            f"Model path: {args.model_path}",
            f"Evaluation output path: {args.evaluation_output}",
        ]

        for line in lines:
            print(line)

        main(args)
