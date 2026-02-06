import numpy as np
import pandas as pd
#Verifying the format of Training Feature Inputs and Target Inputs
def _Xy_checker(X,y):
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
    X_temp=np.array(X)
    y_temp=np.array(y)

    if not np.issubdtype(X_temp.dtype, np.number):
        raise TypeError("X must contain numbers")
    if not np.issubdtype(y_temp.dtype, np.number):
        raise TypeError("y must contain numbers")
    if np.isnan(X_temp).any():
        raise ValueError("There shouldn't be NaN values in X")
    if np.isnan(y_temp).any():
        raise ValueError("There shouldn't be NaN values in y")
    if X_temp.shape[0]!=y_temp.shape[0]:
        raise ValueError(f"X shape is {X_temp.shape}, y shape is {y_temp.shape}")
    return X_temp,y_temp

#Verifying the format of Test Feature Inputs
def _X_t_checker(X_t,X):
    if X_t is None:
        raise ValueError("Inputs can't be none")
    if not(isinstance(X_t,pd.DataFrame) or isinstance(X_t,np.ndarray)):
        raise ValueError("X_t has to be a numpy array or DataFrame")
    if not(X_t.ndim==2 or X_t.ndim==1):
        raise ValueError("X_T has to be 1 or 2 dimensional")
    X_t_temp=np.array(X_t)
    if not np.issubdtype(X_t_temp.dtype, np.number):
        raise TypeError("X_t must contain numbers")
    if np.isnan(X_t_temp).any():
        raise ValueError("There shouldn't be NaN values in X_t")
    if X_t_temp.ndim == 1:
        X_t_temp = X_t_temp.reshape(1, -1)

    if X_t_temp.shape[1] != X.shape[1]:
        raise ValueError("X and X_t must have the same number of features")

    return X_t_temp

#Knn Classifier code
class KnnClassifier:
    def __init__(self,k=None):
        self.k=3 if k==None else k
        self.X=None
        self.y=None
        self.X_t=None
     
    def fit(self,X,y):
        self.X,self.y=_Xy_checker(X,y)
    def predict(self,X_t):
        self.X_t=_X_t_checker(X_t,self.X)
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
        self.X=None
        self.y=None
        self.X_t=None
    
    def fit(self,X,y):
        self.X,self.y=_Xy_checker(X,y)
    def predict(self,X_t):
        self.X_t=_X_t_checker(X_t,self.X)
        results=[]
        for x_t in self.X_t:
            eu_dist = np.sqrt(np.sum((self.X - x_t) ** 2, axis=1))
            idxs = np.argsort(eu_dist)[:self.k]
            results.append(np.mean(self.y[idxs]))
        
        return np.array(results)

        
