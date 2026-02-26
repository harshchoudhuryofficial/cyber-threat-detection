# Cyber Threat Detection using XGBoost

A Machine Learning-based Intrusion Detection System built using the NSL-KDD dataset to classify network traffic as Normal or Attack using XGBoost.

---

## Project Overview

With the rapid increase in cyber attacks, building efficient intrusion detection systems has become critical.

This project develops a high-performance classification model that detects malicious network activity using structured network traffic data.

The model achieves approximately 99.7% accuracy with near-perfect ROC performance.

---

## Problem Statement

Given network connection features, predict whether the traffic is:

- Normal  
- Attack  

The goal is to minimize false positives and false negatives while maintaining high overall accuracy.

---

## Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Joblib  

---

## Project Structure


Cyber-Threat-Detection/
├── KDDTrain+.txt
├── KDDTest+.txt
│── train_model.py
│── requirements.txt
│── README.md
│── .gitignore


---

## Dataset

Dataset used: NSL-KDD

- 41 Features  
- Combination of numerical and categorical attributes  
- Binary classification (Normal vs Attack)  

---

## Data Preprocessing

- Assigned column names  
- Dropped irrelevant difficulty column  
- Converted multi-class labels into binary classification  
- Applied one-hot encoding for categorical variables  
- Used stratified train-test split  

---

## Model Details

Model Used: XGBoost Classifier

Hyperparameters:

- n_estimators = 300  
- max_depth = 8  
- learning_rate = 0.05  
- subsample = 0.8  
- colsample_bytree = 0.8  

Why XGBoost?

- Handles structured tabular data efficiently  
- Reduces bias and variance using boosting  
- High performance for classification tasks  

---

## Model Performance

Confusion Matrix

|                | Predicted Normal | Predicted Attack |
|----------------|-----------------|-----------------|
| Actual Normal  | 15378           | 33              |
| Actual Attack  | 80              | 14213           |

Evaluation Metrics

- Accuracy: ~99.7%  
- Precision: ~99.8%  
- Recall: ~99.4%  
- F1 Score: ~99.6%  
- ROC-AUC Score: 1.000  

---

## Visualizations

- Class Distribution Plot  
- Confusion Matrix  
- ROC Curve  

---

## Model Saving

The trained model is saved as:


cyber_threat_model.pkl


---

## How to Run

Install dependencies:


pip install -r requirements.txt


Run the script:


python train_model.py


---

## Future Improvements

- Hyperparameter tuning using GridSearchCV  
- Feature importance analysis  
- Real-time intrusion detection dashboard  
- Deployment using Flask, FastAPI, or Streamlit  

---

