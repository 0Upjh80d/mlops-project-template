"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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
    parser = argparse.ArgumentParser(
        description="Trains ML model using training dataset. Saves trained model."
    )
    parser.add_argument("--train_data", type=str, help="Path to train dataset.")
    parser.add_argument("--model_output", type=str, help="Path of output model.")

    # Classifier specific arguments
    parser.add_argument(
        "--regressor__n_estimators", type=int, default=500, help="Number of trees."
    )
    parser.add_argument(
        "--regressor__bootstrap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether bootstrap samples are used when building trees.",
    )
    parser.add_argument(
        "--regressor__max_depth",
        type=int,
        default=10,
        help="Maximum number of levels in tree.",
    )
    parser.add_argument(
        "--regressor__max_features",
        default=1.0,
        help="The number of features to consider when looking for the best split.",
    )
    parser.add_argument(
        "--regressor__min_samples_leaf",
        type=int,
        default=4,
        help="Minimum number of samples required at each leaf node.",
    )
    parser.add_argument(
        "--regressor__min_samples_split",
        type=int,
        default=5,
        help="Minimum number of samples required to split a node.",
    )
    parser.add_argument(
        "--regressor__random_state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    """
    Main function that trains and saves the model.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    # Reads the train data
    train_data = pd.read_parquet(Path(args.train_data))

    # Split the data into X and y
    y_train = train_data[TARGET_COL]
    X_train = train_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    # Train a Random Forest Regression Model with the training set
    model = RandomForestRegressor(
        n_estimators=args.regressor__n_estimators,
        bootstrap=args.regressor__bootstrap,
        max_depth=args.regressor__max_depth,
        max_features=args.regressor__max_features,
        min_samples_leaf=args.regressor__min_samples_leaf,
        min_samples_split=args.regressor__min_samples_split,
        random_state=args.regressor__random_state,
    )

    # Log model parameters
    mlflow.log_params(
        {
            "model": "RandomForestRegressor",
            "n_estimators": args.regressor__n_estimators,
            "bootstrap": args.regressor__bootstrap,
            "max_depth": args.regressor__max_depth,
            "max_features": args.regressor__max_features,
            "min_samples_leaf": args.regressor__min_samples_leaf,
            "min_samples_split": args.regressor__min_samples_split,
            "random_state": args.regressor__random_state,
        }
    )

    # Train model with the train set
    model.fit(X_train, y_train)

    # Predict using the Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)

    # Log model performance metrics
    mlflow.log_metrics(
        {"train_r2": r2, "train_mse": mse, "train_rmse": rmse, "train_mae": mae}
    )

    # Visualize results
    plt.scatter(y_train, yhat_train, color="black")
    plt.plot(y_train, y_train, color="blue", linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)


if __name__ == "__main__":

    # Start an MLflow run
    with mlflow.start_run():

        # Parse the input arguments
        args = parse_args()

        lines = [
            f"Train dataset input path: {args.train_data}",
            f"Model output path: {args.model_output}",
            f"n_estimators: {args.regressor__n_estimators}",
            f"bootstrap: {args.regressor__bootstrap}",
            f"max_depth: {args.regressor__max_depth}",
            f"max_features: {args.regressor__max_features}",
            f"min_samples_leaf: {args.regressor__min_samples_leaf}",
            f"min_samples_split: {args.regressor__min_samples_split}",
            f"random_state: {args.regressor__random_state}",
        ]

        for line in lines:
            print(line)

        main(args)
