# Diabetes-Prediction
# Diabetes Prediction Using Machine Learning

## About the Project

This project uses machine learning to predict whether a person is likely to have diabetes based on medical information such as glucose level, blood pressure, BMI, age, and other health-related features.

The goal is to explore the dataset, preprocess the data, train different machine learning models, compare their performance, and save a trained model for future predictions.

## Dataset

The project uses `diabetes.csv`, which contains medical information about patients.

**Input features include:**
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

**Target variable:** `Outcome`

- `0` — Non-Diabetic
- `1` — Diabetic

## Project Workflow

### 1. Data Exploration
- Load the dataset using Pandas.
- Inspect the columns, data types, missing values, and summary statistics.
- Use correlation heatmaps and histograms to understand feature distributions and relationships.

### 2. Data Preprocessing
- Replace selected zero values with statistical values such as the median or mean.
- Separate the input features (`X`) from the target (`y`).
- Detect outliers using the Interquartile Range (IQR) method.
- Standardize the input features using `StandardScaler`.
- Apply an additional filtering step based on the 95th percentile of feature values.

### 3. Train-Test Split
Split the processed dataset into training and testing sets.

- **Training set:** Used to train the models.
- **Testing set:** Used to evaluate their performance.

The code uses a 33% test size and a random state of 42.

### 4. Handle Class Imbalance
Apply **SMOTE (Synthetic Minority Over-sampling Technique)** to the training data to balance the number of samples in the two classes.

This helps the models learn from both diabetic and non-diabetic examples.

### 5. Train Machine Learning Models
The project trains and explores the following classification algorithms:

- **Logistic Regression:** A baseline model for binary classification.
- **Decision Tree:** Learns decision rules from the input features.
- **Random Forest:** Combines multiple decision trees to make predictions.
- **Support Vector Machine (SVM):** Learns a decision boundary between the classes.
- **K-Nearest Neighbors (KNN):** Predicts a class using nearby training samples.

### 6. Model Optimization
Use `GridSearchCV` to search for promising hyperparameters for Random Forest and SVM through cross-validation.

This helps explore model configurations that may improve predictive performance.

### 7. Model Evaluation
Evaluate predictions using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics help assess how well the models distinguish between diabetic and non-diabetic cases.

### 8. Save and Load the Model
Save the trained Logistic Regression model using Python's `pickle` module.

The saved file is named `classification_model.pkl` and can be loaded later to make predictions on appropriately preprocessed input data.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Jupyter Notebook
- Pickle

## How to Run the Project

1. Clone or download this repository.
2. Make sure `diabetes.csv` is in the project directory.
3. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
   ```

4. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `Diabetes_Prediction.ipynb` and run the cells in order.
 .
 

## Author

**Priyanka Choudhary**
