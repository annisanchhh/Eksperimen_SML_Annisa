import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # =========================
    # Load dataset preprocessing
    # =========================
    data_path = "../preprocessing/heart_preprocessing.csv"
    df = pd.read_csv(data_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================
    # Hyperparameter Tuning
    # =========================
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    }

    base_model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # =========================
    # Evaluation
    # =========================
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # =========================
    # Manual MLflow Logging
    # =========================
    mlflow.set_experiment("Heart Disease Classification - Tuning")

    with mlflow.start_run(run_name="RandomForest_Tuning"):
        # log parameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)

        # log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

    print("Tuning selesai")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)


if __name__ == "__main__":
    main()
