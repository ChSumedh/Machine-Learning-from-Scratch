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
    X_temp=np.array(X)
    y_temp=np.array(y)

    if X_temp.ndim!=2:
        raise ValueError("X has to be 2 dimensional")
    if y_temp.ndim!=1:
        raise ValueError("y has to be 1 dimensional")

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
    X_t_temp=np.array(X_t)
    if not(X_t_temp.ndim==2 or X_t_temp.ndim==1):
        raise ValueError("X_t has to be 1 or 2 dimensional")
    if not np.issubdtype(X_t_temp.dtype, np.number):
        raise TypeError("X_t must contain numbers")
    if np.isnan(X_t_temp).any():
        raise ValueError("There shouldn't be NaN values in X_t")
    if X_t_temp.ndim == 1:
        X_t = X_t.reshape(1, -1)

    if X_t_temp.shape[1] != X.shape[1]:
        raise ValueError("X and X_t must have the same number of features")

    return X_t_temp

class LinearRegressor:
    def __init__(self):
        self.X=None
        self.y=None
        self.thetas=None
        self.X_=None
    
    def fit(self,X,y):
        self.X,self.y=_Xy_checker(X,y)
        ones=np.ones((self.X.shape[0],1))
        X_=np.concatenate((ones,self.X),axis=1)
        self.X_=X_
        thetas=np.matmul(np.linalg.pinv(np.matmul(np.transpose(self.X_),self.X_)),
                            np.matmul(np.transpose(self.X_),self.y))
        self.thetas=thetas

    
    def predict(self,X_t):
        X_tn=_X_t_checker(X_t,self.X)
        ones=np.ones((X_tn.shape[0],1))
        X_tn=np.concatenate((ones,X_tn),axis=1)
        result = np.matmul(X_tn,self.thetas)
        return result
                
class SGD_LinearRegressor:
    def __init__(self,batch_size=None,alpha=0.1,epochs=None):
        self.batch_size=batch_size if batch_size!=None else 32
        self.alpha=alpha
        self.epochs=epochs
        self.X=None
        self.y=None
        self.X_=None
        self.thetas=None
    def fit(self,X,y):
        self.X,self.y=_Xy_checker(X,y)
        ones=np.ones((self.X.shape[0],1))
        X_=np.concatenate((ones,X),axis=1)
        self.X_=X_
        self.thetas = np.random.random((self.X.shape[1]+1,))
        max_iter=100_000 if self.epochs==None else self.epochs
        tol=1e-3

        for i in range(max_iter):
            grad=np.zeros(self.X.shape[1]+1)
            k=0

            idx = np.random.permutation(len(X))
            self.X_=self.X_[idx]
            self.y=self.y[idx]
            z=np.matmul(self.X_,self.thetas)
            curr_mse=0.5*np.mean(np.square(self.y-z))

            for batch_start in range(0, len(self.X_), self.batch_size):
                batch_end = batch_start + self.batch_size
                X_batch = self.X_[batch_start:batch_end]
                y_batch = self.y[batch_start:batch_end]

                z_batch = np.matmul(X_batch,self.thetas)
                grad = np.sum((y_batch - z_batch)[:, None] * X_batch, axis=0)
                
                self.thetas += self.alpha * grad / len(y_batch)

            new_mse=0.5*np.mean(np.square(self.y-np.matmul(self.X_,self.thetas)))
            if abs(new_mse-curr_mse)<=tol:
                break
    
    def predict(self,X_t):
        if self.thetas is None:
            raise ValueError("Cannot predict without fit data")
        X_tn=_X_t_checker(X_t,self.X)
        ones = np.reshape(np.ones(X_tn.shape[0]),(-1,1))
        X_tn=np.concatenate((ones,X_tn),axis=1)
        return np.matmul(X_tn,self.thetas)

class LogisticRegression:
    def __init__(self,alpha=0.1,batch_size=32,epochs=None):
        self.alpha=alpha
        self.batch_size=batch_size
        self.epochs=epochs
        self.X=None
        self.y=None
        self.values=None
        self.X_=None
        self.thetas=None
    
    def sigmoid(self,values):
        return 1 / (1 + np.exp(-values))
    
    def fit(self,X,y):
        self.X,self.y=_Xy_checker(X,y)
        values,counts=np.unique(self.y)
        if len(values)!=2:
            raise ValueError("Only Binary Classification allowed")
        self.y=(self.y==values[1]).astype(int)
        self.values=values
        
        ones=np.ones((self.X.shape[0],1))
        X_=np.concatenate((ones,self.X),axis=1)
        self.X_=X_
        self.thetas = np.random.random((self.X.shape[1]+1,))
        max_iter=100_000 if self.epochs==None else self.epochs
        tol=1e-3

        for i in range(max_iter):
            grad=np.zeros(self.X.shape[1]+1)
            k=0

            idx = np.random.permutation(len(self.X))
            self.X_=self.X_[idx]
            self.y=self.y[idx]
            z=np.matmul(self.X_,self.thetas)
            curr_loss= -np.mean(self.y*np.log(self.sigmoid(z)) + (1-self.y)*np.log(1-self.sigmoid(z)))


            for batch_start in range(0, len(self.X_), self.batch_size):
                batch_end = batch_start + self.batch_size
                X_batch = self.X_[batch_start:batch_end]
                y_batch = self.y[batch_start:batch_end]

                z_batch = np.matmul(X_batch,self.thetas)
                grad = np.sum((y_batch - self.sigmoid(z_batch))[:, None] * X_batch, axis=0)
                
                self.thetas += self.alpha * grad / len(y_batch)

            new_loss = -np.mean(self.y*np.log(self.sigmoid(np.matmul(self.X_, self.thetas))) + 
                    (1-self.y)*np.log(1-self.sigmoid(np.matmul(self.X_, self.thetas))))


            if abs(new_loss-curr_loss)<=tol:
                break
        
    
    
    def predict_proba(self,X_t):
        X_tn=_X_t_checker(X_t,self.X)
        ones = np.reshape(np.ones(X_tn.shape[0]),(-1,1))
        X_tn=np.concatenate((ones,X_tn),axis=1)
        return self.sigmoid(np.matmul(X_tn,self.thetas))
    
    def predict(self,X_t,threshold=0.5):
        ans=self.predict_proba(X_t)
        ans= self.values[1] if ans>=threshold else self.values[0]
        return ans
    
