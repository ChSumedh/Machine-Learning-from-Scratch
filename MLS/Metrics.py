import numpy as np
import pandas as pd
def Y_checker(y,y_t):
    if y_t is None or y is None:
        raise ValueError("Inputs can't be none")
    if not(isinstance(y_t ,np.ndarray)):
        raise ValueError("y_t has to be a numpy array")
    if not(isinstance(y,np.ndarray)):
        raise ValueError("y has to be a numpy array")
    if y.shape!=y_t.shape:
        raise ValueError("y and y_t have to have the same shape")
    if np.isnan(y_t).any():
        raise ValueError("There shouldn't be NaN values in y_t")
    if np.isnan(y).any():
        raise ValueError("There shouldn't be NaN values in y")

def confusion_matrix(y,y_t):
    Y_checker(y,y_t)
    unique_y = np.unique(y)
    unique_y_t = np.unique(y_t)
    classes = np.unique(np.concatenate([unique_y, unique_y_t]))
    classes=list(classes)
    cm=np.zeros((len(classes),len(classes)))
    cm=pd.DataFrame(cm,columns=classes,index=classes)

    for i in range(y.shape[0]):
        cm.loc[y[i],y_t[i]]+=1
    return cm

def classification_report(y,y_t):
    unique_y = np.unique(y)
    unique_y_t = np.unique(y_t)
    classes = np.unique(np.concatenate([unique_y, unique_y_t]))
    classes=list(classes)
    cm=confusion_matrix(y,y_t)
    cr=np.zeros((len(classes),2))
    for i in range(len(classes)):
        cr[i][0]=cm.loc[classes[i],classes[i]]/np.sum(cm[cm.columns[i]])
        cr[i][1]=cm.loc[classes[i],classes[i]]/np.sum(cm.loc[classes[i],:])
        cr[i][2]=2/((1/cr[i][0])+(1/cr[i][1]))
    cr=pd.DataFrame(cr,columns=["Precision","Recall","F1-Score"],index=classes)
    return cr

def accuracy_score(y,y_t):
    unique_y = np.unique(y)
    unique_y_t = np.unique(y_t)
    classes = np.unique(np.concatenate([unique_y, unique_y_t]))
    classes=list(classes)
    cm=confusion_matrix(y,y_t)
    sum=np.sum(np.sum(cm))
    num=0
    for i in range(len(classes)):
        num+=cm.loc[classes[i],classes[i]]
    if(sum==0):
        return 0
    return num/sum

    
def rmse(y,y_t):
    Y_checker(y,y_t)
    if not (np.issubdtype(y.dtype,np.number) and np.issubdtype(y_t.dtype,np.number)):
        raise ValueError("The inputs have to be numbers")
    return np.sqrt(np.mean(np.square(y-y_t)))
