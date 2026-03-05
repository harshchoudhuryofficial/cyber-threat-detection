import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

from xgboost import XGBClassifier


print("Step 1: Loading Dataset...")

# Load dataset
train = pd.read_csv("KDDTrain+.txt", header=None)
test = pd.read_csv("KDDTest+.txt", header=None)

# Column names
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

print("Step 2: Preprocessing Data...")

train.drop("difficulty", axis=1, inplace=True)
test.drop("difficulty", axis=1, inplace=True)


train["label"] = (train["label"] != "normal").astype(int)
test["label"] = (test["label"] != "normal").astype(int)

full_data = pd.concat([train, test], ignore_index=True)

X = full_data.drop("label", axis=1)
y = full_data["label"]


X = pd.get_dummies(X)

print("Dataset shape:", X.shape)

print("Step 3: Visualizing Class Distribution...")

plt.figure(figsize=(6,5))
y.value_counts().plot(kind="bar", color=["green","red"])
plt.xticks([0,1],["Normal","Attack"], rotation=0)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

print("Step 4: Splitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


print("Step 5: Training XGBoost Model...")

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

print("Model Training Completed")

print("Step 6: Making Predictions...")

y_pred = model.predict(X_test)

print("\nModel Performance")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

print("Step 7: Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")

plt.xticks([0,1],["Normal","Attack"])
plt.yticks([0,1],["Normal","Attack"])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

for i in range(2):
    for j in range(2):
        plt.text(j,i,cm[i,j],ha="center",va="center")

plt.tight_layout()
plt.show()

print("Step 8: ROC Curve...")

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.tight_layout()
plt.show()


print("Step 9: Saving Model...")

joblib.dump(model, "cyber_threat_model.pkl")

print("Model saved as cyber_threat_model.pkl")

print("Step 10: Sample Prediction...")

sample = X_test.iloc[0:1]
prediction = model.predict(sample)[0]

if prediction == 1:
    print("Prediction: Attack Detected")
else:
    print("Prediction: Normal Traffic")

print("\nProgram Finished Successfully")

