# MLS - Machine Learning from Scratch

## 🚀 Overview

MLS is a custom-built, pip-installable machine learning library developed from scratch using NumPy and Pandas, designed to replicate core functionalities of scikit-learn while providing full transparency into model internals.

### 🔑 Key Highlights
- Implemented 8+ ML algorithms from scratch (classification & regression)
- Built modular preprocessing pipeline (scaling, encoding, imputation, splitting)
- Designed sklearn-like API (`fit`, `predict`, `transform`)
- Packaged as an installable Python library via `pip install git+...`
- **Validated against scikit-learn implementations** to ensure correctness and consistency

### 📊 Validation Against Scikit-learn
Custom implementations were benchmarked against scikit-learn equivalents across multiple datasets:
- Comparable accuracy and RMSE metrics
- Similar decision boundaries and prediction behavior
- Consistent preprocessing transformations

(See comparison visuals below)
<img width="745" height="546" alt="Screenshot From 2026-04-11 16-16-32" src="https://github.com/user-attachments/assets/c2615f7c-3f84-4f80-9dcc-db52d040b1c1" />

<img width="738" height="546" alt="Screenshot From 2026-04-11 16-16-46" src="https://github.com/user-attachments/assets/99b8b132-19f4-4fe4-a722-e5949df52c88" />

<img width="738" height="543" alt="Screenshot From 2026-04-11 16-16-58" src="https://github.com/user-attachments/assets/000ce54b-4d15-4518-b2e0-e454a00dd7af" />

<img width="713" height="543" alt="Screenshot From 2026-04-11 16-17-12" src="https://github.com/user-attachments/assets/df0d04cc-c214-4199-b399-804286e37341" />

<img width="732" height="543" alt="Screenshot From 2026-04-11 16-17-29" src="https://github.com/user-attachments/assets/773923dd-268a-4ca6-ae8f-b18f66c4382e" />

<img width="732" height="543" alt="Screenshot From 2026-04-11 16-17-57" src="https://github.com/user-attachments/assets/ebeb9bdc-15f2-46be-8a8e-3afc86880850" />

<img width="728" height="543" alt="Screenshot From 2026-04-11 16-18-20" src="https://github.com/user-attachments/assets/21a51b50-ae53-49a1-948c-09f55aa885f8" />

<img width="728" height="543" alt="Screenshot From 2026-04-11 16-18-27" src="https://github.com/user-attachments/assets/7e2378f6-1ef8-4473-b1d1-1da35474d9b9" />

<img width="728" height="543" alt="Screenshot From 2026-04-11 16-18-35" src="https://github.com/user-attachments/assets/eadbe36b-5396-4e08-8e13-f6eaa130b624" />

<img width="740" height="543" alt="Screenshot From 2026-04-11 16-18-57" src="https://github.com/user-attachments/assets/26d3d033-
cc90-4558-9a33-0136813a5875" />

<img width="740" height="543" alt="Screenshot From 2026-04-11 16-19-04" src="https://github.com/user-attachments/assets/afa464b2-284f-45ea-bb17-f4c7f9e0ff73" />

## 🚀 Features

### **Supervised Learning**
- **Classification**
  - K-Nearest Neighbors (KNN) Classifier
  - Logistic Regression
  - Naive Bayes Classifier
  - Gaussian Classifier

- **Regression**
  - Linear Regression
  - K-Nearest Neighbors (KNN) Regressor
  - Stochastic Gradient Descent (SGD) Regressor

### **Preprocessing**
- **Scalers**
  - StandardScaler
  - MinMaxScaler

- **Encoders**
  - LabelEncoder (categorical → numeric for targets)
  - OrdinalEncoder (ordered categorical → numeric for features)
  - OneHotEncoder

- **Imputers**
  - SimpleImputer (handle missing values with mean/median/mode/fixed strategies)

- **Data Splitting**
  - train_test_split (with stratification support)

### **Metrics**
- **Classification Metrics**
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (precision, recall, F1-score)

- **Regression Metrics**
  - Root Mean Squared Error (RMSE)

---

## 📦 Installation
```bash
# Install the Package
pip install git+https://github.com/ChSumedh/Machine-Learning-From-Scratch.git

# Install dependencies
pip install numpy pandas
```

---

## 🧭 Module Guide

- `Gaussian.py` → Gaussian Classifier, Naive Bayes Classifier  
- `LinearModels.py` → Linear Regression, Logistic Regression, SGD Regressor  
- `KNeighbours.py` → KNN Classifier, KNN Regressor  
- `PreProcessing.py` → Scalers, Encoders, Imputers  
- `Metrics.py` → Evaluation metrics (Accuracy, RMSE, etc.)  
- `ModelSelection.py` → Train-test split utilities  

## 🔧 Usage Examples

### Logistic Regression Example

```python
from mls.LinearModels import LogisticRegression

# Sample training data
X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
y = [0, 0, 1]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predict
prediction = model.predict([[2.5, 3.5]])

print("Prediction:", prediction)
```

### **8. Preprocessing Pipeline**
```python
from mls.PreProcessing import StandardScaler,MinMaxScaler, LabelEncoder ,OneHotEncoder, SimpleImputer
import numpy as np

# Handle missing values
imputer = SimpleImputer(strat='mean')
X_imputed = imputer.fit_transform(X_train)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
###Or
#####scaler = MinMaxScaler()
#####X_scaled = scaler.fit_transform(X_imputed)

# Encode categorical targets
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_train)
###Or
#####encoder = OneHotEncoder()
#####y_encoded = encoder.fit_transform(y_train)

# Transform test data
X_test_processed = scaler.transform(imputer.transform(X_test))
y_test_encoded = encoder.transform(y_test)
```

### **9. Ordinal Encoding with Custom Order**
```python
from mls.PreProcessing import OrdinalEncoder

# Define ordering for categorical features
order = {
    'temperature': ['cold', 'warm', 'hot'],
    'size': ['S', 'M', 'L', 'XL']
}

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X, order=order)

# Get encoding mapping
print(encoder.get_params())
```

---

## 📊 API Reference

### **Models**
```python
ModelName()
```
Standardizes features by removing mean and scaling to unit variance.

**Methods:**
- `fit(X)` - Trains the model based on the given data
- `predict_proba(Test Data)` - Predicts probablity of each sample being belonging to every possible class.(Only works for classifiers other than KnnClassifier) 
- `predict(Test_Data)` - Predicts the class or value of each test sample

### **StandardScaler**
```python
StandardScaler()
```
Standardizes features by removing mean and scaling to unit variance.

**Methods:**
- `fit(X)` - Compute mean and std
- `transform(X)` - Scale features
- `fit_transform(X)` - Fit and transform in one step
- `get_params()` - Return mean and std for each feature

### **LabelEncoder**
```python
LabelEncoder()
```
Encode categorical labels as integers (sorted alphabetically).

**Methods:**
- `fit(X)` - Learn unique categories
- `transform(X)` - Encode categories to integers
- `inverse_transform(X)` - Decode integers back to categories
- `fit_transform(X)` - Fit and transform
- `get_params()` - Return encoding mapping

### **OrdinalEncoder**
```python
OrdinalEncoder(unknown_map=None)
```
Encode ordered categorical features with user-specified order.

**Parameters:**
- `unknown_map` (int/float, optional): Value to use for unknown categories

**Methods:**
- `fit(X, order)` - Learn categories with specified order
  - `order` (dict): Mapping of column names to ordered lists of categories
- `transform(X)` - Encode categories
- `inverse_transform(X)` - Decode back to categories
- `fit_transform(X, order)` - Fit and transform
- `get_params()` - Return encoding mapping

### **SimpleImputer**
```python
SimpleImputer(strat='mean', fixed_val=None)
```
Fill missing values using various strategies.

**Parameters:**
- `strat` (str): Strategy - 'mean', 'median', 'mode', or 'fixed'
- `fixed_val` (dict, optional): Fixed values per column (required if strat='fixed')

**Methods:**
- `fit(X)` - Learn imputation values
- `transform(X)` - Fill missing values
- `fit_transform(X)` - Fit and transform
- `get_params()` - Return fill values

**Note:** 'mean' and 'median' only work on numeric columns. Categorical columns will be skipped with a warning.

### **split**
```python
    split(X, y, split_size=0.2, stratify_y=False, random_state=None)
```
Split data into training and testing sets.

**Parameters:**
- `X` (DataFrame/array): Features
- `y` (Series/array): Target variable
- `split_size` (float): Proportion for test set (0.0 to 1.0)
- `stratify_y` (bool): Maintain class distribution in splits
- `random_state` (int, optional): Seed for reproducibility

**Returns:**
- `X_train, X_test, y_train, y_test`

---

---

## 🔍 Project Structure
```
mls/
├── Gaussian.py
├── KNeighbours.py
├── LinearModels.py
├── Metrics.py
├── ModelSelection.py
└── PreProcessing.py
```
