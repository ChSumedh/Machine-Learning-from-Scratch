# MLS - Machine Learning from Scratch

A comprehensive machine learning library built from scratch in Python, implementing core ML algorithms and preprocessing tools without relying on scikit-learn.

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
# Clone the repository
git clone https://github.com/ChSumedh/MLS.git
cd MLS

# Install dependencies
pip install numpy pandas
```

---

## 🔧 Usage Examples

### **1. Classification with Gaussian Naive Bayes**
```python
import pandas as pd
from MLS.Gaussian import GaussianClassifier
from MLS.ModelSelection import split
from MLS.Metrics import accuracy_score, confusion_matrix

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = split(X, y, split_size=0.2, stratify_y=True)

# Train model
model = GaussianClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
```

### **2. Regression with Linear Regression**
```python
from MLS.LinearModels import LinearRegressor
from MLS.Metrics import rmse

# Train model
model = LinearRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f"RMSE: {rmse(y_test, y_pred)}")
```

### **3. Preprocessing Pipeline**
```python
from MLS.PreProcessing import StandardScaler, LabelEncoder, SimpleImputer
import numpy as np

# Handle missing values
imputer = SimpleImputer(strat='mean')
X_imputed = imputer.fit_transform(X_train)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Encode categorical targets
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_train)

# Transform test data
X_test_processed = scaler.transform(imputer.transform(X_test))
y_test_encoded = encoder.transform(y_test)
```

### **4. Ordinal Encoding with Custom Order**
```python
from MLS.PreProcessing import OrdinalEncoder

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

### **5. K-Nearest Neighbors Classification**
```python
from MLS.KNeighbours import KnnClassifier

# Train model
model = KnnClassifier(k=5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
from MLS.Metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

## 📊 API Reference

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

## 📝 Design Principles

1. **Sklearn-compatible API**: All estimators follow `.fit()`, `.transform()`, `.predict()` patterns
2. **Input validation**: Comprehensive error checking and informative error messages
3. **Method chaining**: Transformers return `self` from `.fit()` for chaining
4. **Pandas-friendly**: Works seamlessly with DataFrames and Series

## 🙏 Acknowledgments

Built as a learning project to understand machine learning algorithms from first principles. Inspired by scikit-learn's API design.

## 🔍 Project Structure
```
MLS/
├── Gaussian.py
├── KNeighbours.py
├── LinearModels.py
├── Metrics.py
├── ModelSelection.py
└── PreProcessing.py
```