"""
Evaluates trained ML model using test dataset.
Saves predictions, evaluation results and deploy flag.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlflow.tracking import MlflowClient
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_COL = "cost"

NUMERIC_COLS = [
    "distance",
    "dropoff_latitude",
    "dropoff_longitude",
    "passengers",
    "pickup_latitude",
    "pickup_longitude",
    "pickup_weekday",
    "pickup_month",
    "pickup_monthday",
    "pickup_hour",
    "pickup_minute",
    "pickup_second",
    "dropoff_weekday",
    "dropoff_month",
    "dropoff_monthday",
    "dropoff_hour",
    "dropoff_minute",
    "dropoff_second",
]

CAT_NOM_COLS = [
    "store_forward",
    "vendor",
]

CAT_ORD_COLS = []


def parse_args() -> argparse.Namespace:
    """Parses the input arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Evaluates the ML model.")
    parser.add_argument("--model_name", type=str, help="Name of registered model.")
    parser.add_argument("--model_input", type=str, help="Path of input model.")
    parser.add_argument("--test_data", type=str, help="Path to test dataset.")
    parser.add_argument(
        "--evaluation_output", type=str, help="Path of evaluation results."
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="cloud",
        choices=["local", "cloud"],
        help="Local or Cloud runner.",
    )

    args = parser.parse_args()
    return args


def model_evaluation(
    X_test: pd.DataFrame, y_test: pd.Series, model: RegressorMixin, output_path: str
) -> tuple[list, float]:
    """
    Evaluates trained ML model using test dataset.

    Args:
        X_test (pd.DataFrame): The test data.
        y_test (pd.Series): The test labels.
        model (RegressorMixin): The trained model.
        output_path (str): The output path for evaluation results.

    Returns:
        tuple[list, float]: The list of predictions and evaluation score (R2).
    """
    # Get predictions
    yhat_test = model.predict(X_test)

    # Save the output data with feature columns, predicted cost, and actual cost in CSV file
    output_data = X_test.copy()
    output_data["real_label"] = y_test
    output_data["predicted_label"] = yhat_test
    output_data.to_csv(Path(output_path) / "predictions.csv", index=False)

    # Evaluate model performance with the test set
    r2 = r2_score(y_test, yhat_test)
    mse = mean_squared_error(y_test, yhat_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, yhat_test)

    # Print score report to a text file
    (Path(output_path) / "score.txt").write_text(
        f"Scored with the following model:\n{format(model)}"
    )
    with open((Path(output_path) / "score.txt"), "a") as outfile:
        outfile.write(f"Mean squared error: {mse:.2f}\n")
        outfile.write(f"Root mean squared error: {rmse:.2f}\n")
        outfile.write(f"Mean absolute error: {mae:.2f}\n")
        outfile.write(f"Coefficient of determination: {r2:.2f}\n")

    mlflow.log_metrics(
        {"test_r2": r2, "test_mse": mse, "test_rmse": rmse, "test_mae": mae}
    )

    # Visualize results
    plt.scatter(y_test, yhat_test, color="black")
    plt.plot(y_test, y_test, color="blue", linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.title("Comparing Model Predictions to Real values — Test Data")
    plt.savefig("predictions.png")
    mlflow.log_artifact("predictions.png")

    return yhat_test, r2


def model_promotion(
    model_name: str,
    output_path: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    yhat_test: list,
    score: float,
) -> tuple[dict, int]:
    """
    Promotes the best model to production if the new score is
    better than the existing one.

    Args:
        model_name (str): The name of the model.
        output_path (str): The output path for evaluation results.
        X_test (pd.DataFrame): The test data.
        y_test (pd.Series): The test labels.
        yhat_test (list): The list of predictions.
        score (float): The new evaluation score.

    Returns:
        tuple[dict, int]: The dictionary of predictions and deploy flag.
    """
    scores = {}
    predictions = {}

    client = MlflowClient()

    for model_run in client.search_model_versions(f"name='{model_name}'"):
        model_version = model_run.version
        mdl = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        predictions[f"{model_name}:{model_version}"] = mdl.predict(X_test)
        scores[f"{model_name}:{model_version}"] = r2_score(
            y_test, predictions[f"{model_name}:{model_version}"]
        )

    if scores:
        if score >= max(list(scores.values())):
            deploy_flag = 1
        else:
            deploy_flag = 0
    else:
        deploy_flag = 1
    print(f"Deploy flag: {bool(deploy_flag)}")

    with open((Path(output_path) / "deploy_flag"), "w") as outfile:
        outfile.write(f"{int(deploy_flag)}")

    # Add current model score and predictions
    scores["current model"] = score
    predictions["currrent model"] = yhat_test

    perf_comparison_plot = pd.DataFrame(scores, index=["r2 score"]).plot(
        kind="bar", figsize=(15, 10)
    )
    perf_comparison_plot.figure.savefig("perf_comparison.png")
    perf_comparison_plot.figure.savefig(Path(output_path) / "perf_comparison.png")

    mlflow.log_metric("deploy flag", bool(deploy_flag))
    mlflow.log_artifact("perf_comparison.png")

    return predictions, deploy_flag


def main(args: argparse.Namespace) -> None:
    """
    Main function that evaluates the model.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    # Load the test data
    test_data = pd.read_parquet(Path(args.test_data))

    # Split the data into inputs and outputs
    y_test = test_data[TARGET_COL]
    X_test = test_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    # Load the model from input port
    model = mlflow.sklearn.load_model(args.model_input)

    yhat_test, score = model_evaluation(X_test, y_test, model, args.evaluation_output)

    if args.runner.lower() == "cloud":
        predictions, deploy_flag = model_promotion(
            args.model_name, args.evaluation_output, X_test, y_test, yhat_test, score
        )


if __name__ == "__main__":

    # Start an MLflow run
    with mlflow.start_run():

        # Parse the input arguments
        args = parse_args()

        lines = [
            f"Model name: {args.model_name}",
            f"Model path: {args.model_input}",
            f"Test data path: {args.test_data}",
            f"Evaluation output path: {args.evaluation_output}",
        ]

        for line in lines:
            print(line)

        main(args)
