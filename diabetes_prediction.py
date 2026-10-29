# -*- coding: utf-8 -*-

# ============================================================
# DIABETES PREDICTION USING MACHINE LEARNING
# ============================================================

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

dataframe = pd.read_csv("diabetes.csv")

print("\nFirst five rows:")
print(dataframe.head())

print("\nColumn names:")
print(dataframe.columns)

print("\nMissing values:")
print(dataframe.isnull().sum())

print("\nData types:")
print(dataframe.dtypes)

print("\nDataset shape:")
print(dataframe.shape)

print("\nSummary statistics:")
print(dataframe.describe())


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

plt.figure(figsize=(12, 10))
sns.heatmap(dataframe.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

dataframe[features].hist(figsize=(12, 10), bins=20)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# Visualize possible outliers.
# Extreme values are not automatically deleted because
# they may represent genuine medical observations.

X = dataframe.drop(columns=["Outcome"])
y = dataframe["Outcome"]

plt.figure(figsize=(12, 8))
sns.boxplot(data=X)
plt.title("Boxplot of Input Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxPlot.jpg")
plt.show()


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())


# ============================================================
# 4. PREPROCESSING FUNCTION
# ============================================================

# In this dataset, zero can represent a missing-like
# measurement for these particular medical features.
# Zero pregnancies is valid, so it is NOT replaced.

zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


def replace_invalid_zeros(data):
    data = data.copy()

    for column in zero_as_missing:
        data[column] = data[column].replace(0, np.nan)

    return data


# ============================================================
# 5. CREATE A REUSABLE ML PIPELINE
# ============================================================

def create_pipeline(model):
    return Pipeline([
        (
            "zero_handling",
            FunctionTransformer(
                replace_invalid_zeros,
                validate=False
            )
        ),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])


# SMOTE is inside the pipeline. During cross-validation,
# it is applied only to each fold's training portion.


# ============================================================
# 6. LOGISTIC REGRESSION
# ============================================================

print("\n========== LOGISTIC REGRESSION ==========")

classification = create_pipeline(LogisticRegression(max_iter=1000))

classification.fit(X_train, y_train)

y_predictions = classification.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_predictions,
    target_names=["Non-Diabetic", "Diabetic"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predictions))


# ============================================================
# 7. DECISION TREE
# ============================================================

print("\n========== DECISION TREE ==========")

dt_pipeline = create_pipeline(
    DecisionTreeClassifier(random_state=42)
)

dt_pipeline.fit(X_train, y_train)

y_pred_dt = dt_pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_dt))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_dt,
    target_names=["Non-Diabetic", "Diabetic"],
    zero_division=0
))

plt.figure(figsize=(18, 10))
plot_tree(
    dt_pipeline.named_steps["model"],
    feature_names=X.columns,
    class_names=["Non-Diabetic", "Diabetic"],
    filled=True,
    max_depth=3
)
plt.title("Decision Tree (First Three Levels)")
plt.show()


# ============================================================
# 8. RANDOM FOREST
# ============================================================

print("\n========== RANDOM FOREST ==========")

rf_pipeline = create_pipeline(
    RandomForestClassifier(random_state=42)
)

rf_param_grid = {
    "model__n_estimators": [50, 100, 120],
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [2, 3, 4, 5, None]
}

grid_search_rf = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search_rf.fit(X_train, y_train)

rf_new = grid_search_rf.best_estimator_

print("Best Random Forest Parameters:")
print(grid_search_rf.best_params_)

y_pred_rf = rf_new.predict(X_test)

print("Random Forest Test Accuracy:",
      accuracy_score(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_rf,
    target_names=["Non-Diabetic", "Diabetic"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))


# ============================================================
# 9. SUPPORT VECTOR MACHINE (SVM)
# ============================================================

print("\n========== SUPPORT VECTOR MACHINE ==========")

svm_pipeline = create_pipeline(SVC())

svm_param_grid = [
    {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear"]
    },
    {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["rbf"],
        "model__gamma": ["scale", "auto"]
    },
    {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["poly"],
        "model__gamma": ["scale", "auto"],
        "model__degree": [2, 3]
    }
]

grid_search_svm = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search_svm.fit(X_train, y_train)

svm_optimized = grid_search_svm.best_estimator_

print("Best SVM Parameters:")
print(grid_search_svm.best_params_)

y_pred_svm = svm_optimized.predict(X_test)

print("Optimized SVM Test Accuracy:",
      accuracy_score(y_test, y_pred_svm))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_svm,
    target_names=["Non-Diabetic", "Diabetic"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))


# ============================================================
# 10. K-NEAREST NEIGHBORS (KNN)
# ============================================================

print("\n========== K-NEAREST NEIGHBORS ==========")

knn_pipeline = create_pipeline(KNeighborsClassifier())

knn_param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11],
    "model__weights": ["uniform", "distance"]
}

grid_search_knn = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=knn_param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search_knn.fit(X_train, y_train)

knn_optimized = grid_search_knn.best_estimator_

print("Best KNN Parameters:")
print(grid_search_knn.best_params_)

y_pred_knn = knn_optimized.predict(X_test)

print("Optimized KNN Test Accuracy:",
      accuracy_score(y_test, y_pred_knn))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_knn,
    target_names=["Non-Diabetic", "Diabetic"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))


# ============================================================
# 11. COMPARE MODELS
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Store predictions from all five models
results = {
    "Logistic Regression": (classification, y_predictions),
    "Decision Tree": (dt_pipeline, y_pred_dt),
    "Random Forest": (rf_new, y_pred_rf),
    "SVM": (svm_optimized, y_pred_svm),
    "KNN": (knn_optimized, y_pred_knn)
}

comparison = []

for model_name, (model, predictions) in results.items():

    # Probability that a patient belongs to class 1 (Diabetic)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]

    # For models without predict_proba, use decision scores
    else:
        y_scores = model.decision_function(X_test)

    comparison.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_scores)
    })

comparison_df = pd.DataFrame(comparison)

comparison_df = comparison_df.sort_values(
    by="ROC-AUC", ascending=False
)

print(comparison_df.round(3).to_string(index=False))

# 12. save the logistic regression model
import pickle

# Save the complete Logistic Regression pipeline
with open("classification_model.pkl", "wb") as file:
    pickle.dump(classification, file)

print("Logistic Regression model saved successfully!")



# ============================================================
# 13. LOAD THE SAVED MODEL AND MAKE PREDICTIONS
# ============================================================

with open("classification_model.pkl", "rb") as file:
    classification_model = pickle.load(file)

loaded_predictions = classification_model.predict(X_test)

print("\nPredictions from the loaded model:")
print(loaded_predictions[:10])

print("\nProject execution completed.")
