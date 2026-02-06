import pandas as pd
import numpy as np

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
    if np.isnan(X_temp).any():
        raise ValueError("There shouldn't be NaN values in X")
    if np.isnan(y_temp).any():
        raise ValueError("There shouldn't be NaN values in y")
    if X_temp.shape[0]!=y_temp.shape[0]:
        raise ValueError(f"X shape is {X_temp.shape}, y shape is {y_temp.shape}")
    return X_temp,y_temp

def split(X,y,split_size,stratify_y,random_state=None):
    if not isinstance(stratify_y,bool):
        raise ValueError("Stratify_y has to be a boolean value")
    Xn,yn=_Xy_checker(X,y)
    rng=np.random.default_rng(random_state if random_state is not None else np.random.randint(100_000_000))
    idx=rng.permutation(Xn.shape[0])
    Xn=Xn[idx]
    yn=yn[idx]

    classes=list(np.unique(yn))
    class_indices={clas:list(np.where(yn==clas)) for clas in classes}

    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]

    if not stratify_y:
        X_test=Xn[0:int((1-split_size)*len(Xn))]
        y_test=yn[0:int((1-split_size)*len(yn))]
        X_train=Xn[int((1-split_size)*len(Xn)):]
        y_train=yn[int((1-split_size)*len(yn)):]

    else:

        for clas in class_indices.keys():
            splitter=int((1-split_size)*len(class_indices[clas][0]))
            X_test.append(Xn[class_indices[clas][0][:splitter]])
            X_train.append(Xn[class_indices[clas][0][splitter:]])
            y_test.append(yn[class_indices[clas][0][:splitter]])
            y_train.append(yn[class_indices[clas][0][splitter:]])
        
        X_train=np.concatenate(X_train)
        y_train=np.concatenate(y_train)
        X_test=np.concatenate(X_test)
        y_test=np.concatenate(y_test)

    return X_train,y_train,X_test,y_test