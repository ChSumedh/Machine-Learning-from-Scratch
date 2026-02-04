import pandas as pd
import numpy as np

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
    if np.isnan(X).any():
        raise ValueError("There shouldn't be NaN values in X")
    if np.isnan(y).any():
        raise ValueError("There shouldn't be NaN values in y")
    if X.shape[0]!=y.shape[0]:
        raise ValueError(f"X shape is {X.shape}, y shape is {y.shape}")
    return X,y

def split(X,y,split_size,stratify_y,random_state=None):
    if not isinstance(stratify_y,bool):
        raise ValueError("Stratify_y has to be a boolean value")
    Xy_checker(X,y)
    rng=np.random.default_rng(random_state if random_state is not None else np.random.randint(100_000_000))
    idx=rng.permutation(X.shape[0])
    X=X[idx]
    y=y[idx]

    classes=list(np.unique(y))
    class_indices={clas:list(np.where(y==clas)) for clas in classes}

    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]

    if not stratify_y:
        X_test=X[0:int((1-split_size)*len(X))]
        y_test=y[0:int((1-split_size)*len(y))]
        X_train=X[int((1-split_size)*len(X)):]
        y_train=y[int((1-split_size)*len(y)):]

    else:

        for clas in class_indices.keys():
            splitter=int((1-split_size)*len(class_indices[clas][0]))
            X_test.append(X[class_indices[clas][0][:splitter]])
            X_train.append(X[class_indices[clas][0][splitter:]])
            y_test.append(y[class_indices[clas][0][:splitter]])
            y_train.append(y[class_indices[clas][0][splitter:]])
        
        X_train=np.concatenate(X_train)
        y_train=np.concatenate(y_train)
        X_test=np.concatenate(X_test)
        y_test=np.concatenate(y_test)

    return X_train,y_train,X_test,y_test