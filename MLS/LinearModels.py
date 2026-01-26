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
    X=np.array(X)
    y=np.array(y)

    if X.ndim!=2:
        raise ValueError("X has to be 2 dimensional")
    if y.ndim!=1:
        raise ValueError("y has to be 1 dimensional")

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
    X_t=np.array(X_t)
    if not(X_t.ndim==2 or X_t.ndim==1):
        raise ValueError("X_t has to be 1 or 2 dimensional")
    if not np.issubdtype(X_t.dtype, np.number):
        raise TypeError("X_t must contain numbers")
    if np.isnan(X_t).any():
        raise ValueError("There shouldn't be NaN values in X_t")
    if X_t.ndim == 1:
        X_t = X_t.reshape(1, -1)

    if X_t.shape[1] != X.shape[1]:
        raise ValueError("X and X_t must have the same number of features")

    return X_t

class LinearRegressor:
    def fit(self,X,y):
        X,y=Xy_checker(X,y)
        self.X=X
        self.y=y
        ones=np.ones((X.shape[0],1))
        X_=np.concatenate((ones,X),axis=1)
        self.X_=X_
        thetas=np.matmul(np.linalg.pinv(np.matmul(np.transpose(self.X_),self.X_)),
                            np.matmul(np.transpose(self.X_),self.y))
        self.thetas=thetas

    
    def predict(self,X_t):
        X_t=X_t_checker(X_t,self.X)
        ones=np.ones((X_t.shape[0],1))
        X_t=np.concatenate((ones,X_t),axis=1)
        result = np.matmul(X_t,self.thetas)
        return result
                
class SGD_LinearRegressor:
    def __init__(self,batch_size=None,alpha=0.1,epochs=None):
        self.batch_size=batch_size if batch_size!=None else 32
        self.alpha=alpha
        self.epochs=epochs
    def fit(self,X,y):
        X,y=Xy_checker(X,y)
        self.X=X
        self.y=y
        ones=np.ones((X.shape[0],1))
        X_=np.concatenate((ones,X),axis=1)
        self.X_=X_
        self.thetas = np.random.random((X.shape[1]+1,))
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
        X_t=X_t_checker(X_t,self.X)
        ones = np.reshape(np.ones(X_t.shape[0]),(-1,1))
        X_t=np.concatenate((ones,X_t),axis=1)
        return np.matmul(X_t,self.thetas)

class LogisticRegression:
    def __init__(self,alpha=0.1,batch_size=32,epochs=None):
        self.alpha=alpha
        self.batch_size=batch_size
        self.epochs=epochs
    
    def sigmoid(self,values):
        return 1 / (1 + np.exp(-values))
    
    def fit(self,X,y):
        X,y=Xy_checker(X,y)
        self.X=X
        self.y=y
        values,counts=np.unique(y)
        if len(values)!=2:
            raise ValueError("Only Binary Classification allowed")
        y=(y==values[1]).astype(int)
        self.y=y
        self.values=values
        
        ones=np.ones((X.shape[0],1))
        X_=np.concatenate((ones,X),axis=1)
        self.X_=X_
        self.thetas = np.random.random((X.shape[1]+1,))
        max_iter=100_000 if self.epochs==None else self.epochs
        tol=1e-3

        for i in range(max_iter):
            grad=np.zeros(self.X.shape[1]+1)
            k=0

            idx = np.random.permutation(len(X))
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
        X_t_checker(X_t,self.X)
        ones = np.reshape(np.ones(X_t.shape[0]),(-1,1))
        X_t=np.concatenate((ones,X_t),axis=1)
        return self.sigmoid(np.matmul(X_t,self.thetas))
    
    def predict(self,X_t,threshold=0.5):
        ans=self.predict_proba(X_t)
        ans= self.values[1] if ans>=threshold else self.values[0]
        return ans
    
