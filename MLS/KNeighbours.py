import numpy as np
import pandas as pd
#Verifying the format of Training Feature Inputs and Target Inputs
def Xy_checker(X,y):
    if X is None or y is None:
        raise ValueError("Inputs can't be none")
    if not(isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)):
        raise ValueError("X has to be a numpy array or DataFrame")
    if not(isinstance(y,pd.DataFrame) or isinstance(y,np.ndarray)):
        raise ValueError("y has to be a numpy array or DataFrame")
    if X.ndim!=2:
        raise ValueError("X has to be 2 dimensional")
    if y.ndim!=1:
        raise ValueError("y has to be 1 dimensional")
    X=np.array(X)
    y=np.array(y)

    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("X must contain numbers")
    if not np.issubdtype(y.dtype, np.number):
        raise TypeError("y must contain numbers")
    if np.isnan(X).any():
        raise ValueError("There shouldn't be NaN values in X")
    if np.isnan(y).any():
        raise ValueError("There shouldn't be NaN values in y")
    if X.shape[0]!=y.shape[0]:
        raise ValueError(f"X shape is {X.shape}, y shape is {y.shape}")
    return X,y

#Verifying the format of Test Feature Inputs
def X_t_checker(X_t,X):
    if X_t is None:
        raise ValueError("Inputs can't be none")
    if not(isinstance(X_t,pd.DataFrame) or isinstance(X_t,np.ndarray)):
        raise ValueError("X_t has to be a numpy array or DataFrame")
    if not(X_t.ndim==2 or X_t.ndim==1):
        raise ValueError("X_T has to be 1 or 2 dimensional")
    X_t=np.array(X_t)
    if not np.issubdtype(X_t.dtype, np.number):
        raise TypeError("X_t must contain numbers")
    if np.isnan(X_t).any():
        raise ValueError("There shouldn't be NaN values in X_t")
    if X_t.ndim == 1:
        X_t = X_t.reshape(1, -1)

    if X_t.shape[1] != X.shape[1]:
        raise ValueError("X and X_t must have the same number of features")

    return X_t

#Knn Classifier code
class KnnClassifier:
    def __init__(self,k=None):
        self.k=3 if k==None else k
     
    def fit(self,X,y):
        X,y=Xy_checker(X,y)
        self.X=X
        self.y=y
    def predict(self,X_t):
        X_t=X_t_checker(X_t,self.X)
        self.X_t=X_t
        results=[]
        for x_t in self.X_t:
            eu_dist = np.sqrt(np.sum((self.X - x_t) ** 2, axis=1))
            idxs = np.argsort(eu_dist)[:self.k]
    
            class_counts = {}
            for i in idxs:
                weight = 1 / (eu_dist[i] + 1e-9)
                cls = self.y[i]
                class_counts[cls] = class_counts.get(cls, 0) + weight

            results.append(max(class_counts, key=class_counts.get))
        return np.array(results)
    
#Knn Regressor
class KnnRegressor:
    def __init__(self,k=None):
        self.k=3 if k==None else k
    
    def fit(self,X,y):
        X,y=Xy_checker(X,y)
        self.X=X
        self.y=y
    def predict(self,X_t):
        X_t=X_t_checker(X_t,self.X)
        self.X_t=X_t
        results=[]
        for x_t in self.X_t:
            eu_dist = np.sqrt(np.sum((self.X - x_t) ** 2, axis=1))
            idxs = np.argsort(eu_dist)[:self.k]
            results.append(np.mean(self.y[idxs]))
        
        return np.array(results)

        
