import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # =====================
    # Load dataset (PREPROCESSED)
    # =====================
    data_path = "../preprocessing/heart_preprocessing.csv"
    df = pd.read_csv(data_path)

    # =====================
    # Split features & target
    # =====================
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =====================
    # MLflow autolog
    # =====================
    mlflow.set_experiment("Heart Disease Classification")

    with mlflow.start_run():
        mlflow.sklearn.autolog(log_models=False)
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        # =====================
        # Evaluation
        # =====================
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy :", acc)
        print("Precision:", prec)
        print("Recall   :", rec)
        print("F1-score :", f1)


if __name__ == "__main__":
    main()
