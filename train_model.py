"""Cyber threat detection ML pipeline."""

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

from xgboost import XGBClassifier


def load_data():
    train = pd.read_csv("KDDTrain+.txt", header=None)
    test = pd.read_csv("KDDTest+.txt", header=None)
    return train, test


def preprocess_data(train, test):
    col_names = [...]  # keep same list

    train.columns = col_names
    test.columns = col_names

    train.drop("difficulty", axis=1, inplace=True)
    test.drop("difficulty", axis=1, inplace=True)

    train["label"] = (train["label"] != "normal").astype(int)
    test["label"] = (test["label"] != "normal").astype(int)

    full_data = pd.concat([train, test], ignore_index=True)

    X = pd.get_dummies(full_data.drop("label", axis=1))
    y = full_data["label"]

    return X, y


def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nModel Performance")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

    return y_pred


def main():
    train, test = load_data()
    X, y = preprocess_data(train, test)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    joblib.dump(model, "cyber_threat_model.pkl")


if __name__ == "__main__":
    main()
