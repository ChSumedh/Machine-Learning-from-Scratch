# MLS - Machine Learning from Scratch

A comprehensive machine learning library built from scratch in Python, implementing core ML algorithms and preprocessing tools without relying on scikit-learn.

## 🚀 Features

### **Supervised Learning**
- **Classification**
  - K-Nearest Neighbors (KNN) Classifier
  - Logistic Regression
  - Naive Bayes Classifier
  - Gaussian Naive Bayes

- **Regression**
  - Linear Regression
  - K-Nearest Neighbors (KNN) Regressor
  - Stochastic Gradient Descent (SGD) Regressor

### **Preprocessing**
- **Scalers**
  - StandardScaler (mean=0, std=1 normalization)
  - MinMaxScaler (planned)

- **Encoders**
  - LabelEncoder (categorical → numeric for targets)
  - OrdinalEncoder (ordered categorical → numeric for features)
  - OneHotEncoder (planned)

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
git clone https://github.com/yourusername/MLS.git
cd MLS

# Install dependencies
pip install numpy pandas --break-system-packages
```

---

## 🔧 Usage Examples

### **1. Classification with Gaussian Naive Bayes**
```python
import pandas as pd
from MLS.naive_bayes import GaussianNB
from MLS.preprocessing import train_test_split
from MLS.metrics import accuracy_score, confusion_matrix

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, split_size=0.2, stratify_y=True)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
```

### **2. Regression with Linear Regression**
```python
from MLS.linear_regression import LinearRegression
from MLS.metrics import rmse

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f"RMSE: {rmse(y_test, y_pred)}")
```

### **3. Preprocessing Pipeline**
```python
from MLS.preprocessing import StandardScaler, LabelEncoder, SimpleImputer
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
from MLS.preprocessing import OrdinalEncoder

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
from MLS.knn import KNNClassifier

# Train model
model = KNNClassifier(k=5, distance_metric='euclidean')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
from MLS.metrics import classification_report
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

### **train_test_split**
```python
train_test_split(X, y, split_size=0.2, stratify_y=False, random_state=None)
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

## 🎯 Roadmap

### **High Priority**
- [ ] Decision Trees (classification & regression)
- [ ] Random Forest
- [ ] Cross-validation (k-fold, stratified)
- [ ] Pipeline class
- [ ] MinMaxScaler
- [ ] OneHotEncoder
- [ ] Additional metrics (ROC-AUC, R², MAE)

### **Medium Priority**
- [ ] Support Vector Machines (SVM)
- [ ] Ridge/Lasso Regression
- [ ] GridSearchCV
- [ ] K-Means Clustering
- [ ] PCA (dimensionality reduction)
- [ ] Ensemble methods (Bagging, AdaBoost)

### **Lower Priority**
- [ ] Gradient Boosting
- [ ] Neural Networks
- [ ] Feature selection tools
- [ ] Time series models

---

## 🧪 Testing
```python
# Example test script
import numpy as np
from MLS.preprocessing import StandardScaler

# Create test data
X = np.array([[1, 2], [3, 4], [5, 6]])

# Test scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original:", X)
print("Scaled:", X_scaled)
print("Mean:", X_scaled.mean(axis=0))  # Should be ~[0, 0]
print("Std:", X_scaled.std(axis=0))    # Should be ~[1, 1]
```

---

## 📝 Design Principles

1. **Sklearn-compatible API**: All estimators follow `.fit()`, `.transform()`, `.predict()` patterns
2. **Input validation**: Comprehensive error checking and informative error messages
3. **Method chaining**: Transformers return `self` from `.fit()` for chaining
4. **Pandas-friendly**: Works seamlessly with DataFrames and Series
5. **No black boxes**: Clean, readable implementations for educational purposes

---

## 🤝 Contributing

Contributions are welcome! Areas that need help:
- Implementing remaining algorithms
- Adding unit tests
- Improving documentation
- Performance optimizations
- Bug fixes

---

## 📄 License

MIT License - feel free to use this code for learning and projects!

---

## 🙏 Acknowledgments

Built as a learning project to understand machine learning algorithms from first principles. Inspired by scikit-learn's API design.

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

## 🔍 Project Structure
```
MLS/
├── classification/
│   ├── knn.py
│   ├── logistic.py
│   ├── naive_bayes.py
│   └── gaussian.py
├── regression/
│   ├── linear.py
│   ├── knn.py
│   └── sgd.py
├── preprocessing/
│   ├── scalers.py (StandardScaler)
│   ├── encoders.py (LabelEncoder, OrdinalEncoder)
│   ├── imputers.py (SimpleImputer)
│   └── split.py (train_test_split)
├── metrics/
│   ├── classification.py (accuracy_score, confusion_matrix, classification_report)
│   └── regression.py (rmse)
└── README.md
```

---

**Happy Learning! 🎓**