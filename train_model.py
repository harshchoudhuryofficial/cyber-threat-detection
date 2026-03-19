"""Cyber threat detection ML pipeline."""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from xgboost import XGBClassifier


def load_data():
    """Load training and testing datasets."""
    train = pd.read_csv("KDDTrain+.txt", header=None)
    test = pd.read_csv("KDDTest+.txt", header=None)
    return train, test


def preprocess_data(train, test):
    """Preprocess datasets and return features and labels."""
    col_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label", "difficulty"
    ]

    train.columns = col_names
    test.columns = col_names

    train.drop("difficulty", axis=1, inplace=True)
    test.drop("difficulty", axis=1, inplace=True)

    train["label"] = (train["label"] != "normal").astype(int)
    test["label"] = (test["label"] != "normal").astype(int)

    full_data = pd.concat([train, test], ignore_index=True)

    features = pd.get_dummies(full_data.drop("label", axis=1))
    target = full_data["label"]

    return features, target


def train_model(features_train, target_train):
    """Train the XGBoost model."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(features_train, target_train)
    return model


def evaluate_model(model, features_test, target_test):
    """Evaluate model performance."""
    predictions = model.predict(features_test)

    print("\nModel Performance")
    print(f"Accuracy: {accuracy_score(target_test, predictions):.2f}")
    print(f"Precision: {precision_score(target_test, predictions):.2f}")
    print(f"Recall: {recall_score(target_test, predictions):.2f}")
    print(f"F1 Score: {f1_score(target_test, predictions):.2f}")


def main():
    """Main execution function."""
    train, test = load_data()

    features, target = preprocess_data(train, test)

    features_train, features_test, target_train, target_test = train_test_split(
        features, target,
        test_size=0.2,
        stratify=target,
        random_state=42
    )

    model = train_model(features_train, target_train)

    evaluate_model(model, features_test, target_test)

    joblib.dump(model, "cyber_threat_model.pkl")
    print("Model saved successfully")


if __name__ == "__main__":
    main()
