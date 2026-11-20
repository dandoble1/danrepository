import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
# Enable automatic MLflow tracking
mlflow.autolog(log_models=True)
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Model parameters
n_estimators = 100
max_depth = 5
# Create model
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)
# Train model
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="weighted"
)
recall = recall_score(
    y_test, y_pred, average="weighted"
)
f1 = f1_score(
    y_test, y_pred, average="weighted"
)
# Log custom metrics
mlflow.log_metric("test_accuracy", accuracy)
mlflow.log_metric("test_precision", precision)
mlflow.log_metric("test_recall", recall)
mlflow.log_metric("test_f1_score", f1)
# Log additional parameters
mlflow.log_param("dataset", "Iris")
mlflow.log_param("algorithm", "Random Forest")
mlflow.log_param("test_size", 0.2)
print("Model Training Completed!")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))
