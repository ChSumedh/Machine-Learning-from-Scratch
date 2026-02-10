import numpy as np
import pandas as pd

def _Xy_checker(X,y):
    if X is None or y is None:
        raise ValueError("Inputs can't be none")
    if not(isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)):
        raise ValueError("X has to be a numpy array or DataFrame")
    if not(isinstance(y,pd.Series) or isinstance(y,np.ndarray)):
        raise ValueError("y has to be a numpy array or Series")
    X_temp=pd.DataFrame(X)
    y_temp=pd.Series(y)
    if X_temp.ndim!=2:
        raise ValueError("X has to be 2 dimensional")
    if y_temp.ndim!=1:
        raise ValueError("y has to have only one column")
    if X_temp.isna().any().any():
        raise ValueError("There shouldn't be NaN values in X")
    if y_temp.isna().any():
        raise ValueError("There shouldn't be NaN values in y")
    if X_temp.shape[0]!=y_temp.shape[0]:
        raise ValueError(f"X shape is {X_temp.shape}, y shape is {y_temp.shape}")
    return X_temp,y_temp

def _X_t_checker(X_t,X):
    if X_t is None:
        raise ValueError("Inputs can't be none")
    if not(isinstance(X_t,pd.DataFrame) or isinstance(X_t,np.ndarray)):
        raise ValueError("X_t has to be a numpy array or DataFrame")
    if isinstance(X_t,np.ndarray):
        if not(X_t.ndim==2 or X_t.ndim==1):
            raise ValueError("X_T has to be 1 or 2 dimensional")
        X_t_temp=pd.DataFrame(X_t)
    else:
        X_t_temp=X_t
    if X_t_temp.isna().any().any():
        raise ValueError("There shouldn't be NaN values in X_t")
    if X_t_temp.shape[1] != X.shape[1]:
        raise ValueError("X and X_t must have the same number of features")
    for col in X.columns:
        if col not in X_t_temp.columns:
            raise ValueError(f"Missing column in X_t: {col}")

        x_dtype = X[col]
        xt_dtype = X_t_temp[col]

        if pd.api.types.is_numeric_dtype(x_dtype) != pd.api.types.is_numeric_dtype(xt_dtype):
            raise TypeError(
                f"Dtype mismatch in column '{col}': "
                f"train={x_dtype.dtype}, test={xt_dtype.dtype}"
            )

    return X_t_temp

class NaiveBayesClassifier:

    def __init__(self):
        self.X=None
        self.y=None
        self.numerical=None
        self.categorical=None
        self.classes=None
        self.prior_proba=None

    @staticmethod
    def normal_likelihood(val,mean,std):
        if std==0:
            return 1e-9
        return max((1/(np.sqrt(2*np.pi)*std))*(np.e**(-0.5*(np.square((val-mean)/std)))),10e-9)
    
    def fit(self,X,y):
        self.X,self.y=_Xy_checker(X,y)
        numerical={}
        categorical={}
        classes=list(self.y.unique())
        class_indices={clas:self.y.loc[(self.y==clas)].index for clas in classes }

        #Caluculating Prior Probablities
        prior_proba = self.y.iloc[:, 0].value_counts(normalize=True).to_dict()
        
        for clas,index in class_indices.items():
            curr = self.X.iloc[index]
            for col in curr.columns:
                if pd.api.types.is_numeric_dtype(curr[col]):
                    numerical.setdefault(clas,{})
                    numerical[clas][col]=[curr[col].mean(),curr[col].std()+1e-9]
                else:
                    categorical.setdefault(clas, {})
                    categorical[clas][col]={val:0 for val in self.X[col].unique()}
                    counts = curr[col].value_counts()
                    categorical[clas][col] = (counts / counts.sum()).to_dict()

        self.numerical=numerical
        self.categorical=categorical
        self.classes=list(class_indices.keys())
        self.prior_proba=prior_proba

    def predict_proba(self,X_t):
        if self.classes is None:
            raise ValueError("Cannot predict without training data")
        X_tn=_X_t_checker(X_t,self.X)
        probs=np.zeros((X_tn.shape[0],len(self.classes)))

        k=0
        for clas in self.classes:
            for i in range(0,X_tn.shape[0]):
                log_t=0
                for col in X_tn.columns:
                    if pd.api.types.is_numeric_dtype(X_tn[col]):
                        log_t+=np.log((self.normal_likelihood(X_tn.loc[i,col],self.numerical[clas][col][0],
                                                    self.numerical[clas][col][1])))
                    else:
                        log_t+=np.log(self.categorical[clas][col].get(X_tn.loc[i, col], 1e-9))
                probs[i][k]=log_t+np.log(self.prior_proba[clas])
                probs[i][k]=np.exp(probs[i][k])+1e-9
            k=k+1
        
        for i in range(0,probs.shape[0]):
            div=np.sum(probs[i])
            for j in range(0,probs.shape[1]):
                if div>0:
                    probs[i][j]/=div
        
        probs=pd.DataFrame(probs,columns=self.classes)
        return probs
    
    def predict(self,X_t):
        probs=np.array(self.predict_proba(X_t))
        ans=[]
        classes = list(self.classes)

        for i in range(probs.shape[0]):
            ans.append(classes[np.argmax(probs[i])])
        ans = np.array(ans)
        ans=ans.reshape((-1,))

        return ans

class GaussianClassifier:

    def __init__(self):
        self.X=None
        self.y=None
        self.classes=None
        self.sigmas=None
        self.means=None

    @staticmethod
    def normal_likelihood(val,mean,sigma):
        if np.linalg.det(sigma) ==0:
            return 1e-9
        return (1/(np.pow(2*np.pi,0.5)*np.sqrt(np.linalg.det(sigma))))*(np.exp(
            -0.5*np.matmul(np.matmul(np.transpose(val-mean), np.linalg.pinv(sigma)), (val-mean))
))
    
    def fit(self,X,y):
        self.X,self.y=_Xy_checker(X,y)
        if not self.X.select_dtypes(include=[np.number]).shape[1] == self.X.shape[1]:
            raise TypeError("X must contain numbers")
        if not self.y.select_dtypes(include=[np.number]).shape[1] == self.y.shape[1]:
            raise TypeError("y must contain numbers")
        means={}
        classes=list(self.y.unique())
        self.classes=classes
        class_indices={clas:(self.y.loc[self.y==clas]).index for clas in classes }
        for clas,index in class_indices.items():
            curr=self.X.iloc[index]
            means[clas]=list(curr.mean())
        sigmas={}
        for clas in classes:
            class_data = self.X.iloc[class_indices[clas]]
            m = np.array(means[clas])
            centered = class_data.values - m
            sigmas[clas] = (centered.T @ centered) / len(class_data)
        self.sigmas=sigmas
        self.means=means
    
    def predict_proba(self,X_t):
        if self.sigma is None:
            raise ValueError("Cannot predict without training data")
        X_tn=_X_t_checker(X_t,self.X)
        probas=[]
        for i in range(len(X_tn)):
            proba=[]
            for clas in self.classes:
                proba.append(self.normal_likelihood(X_tn.iloc[i],self.means[clas],self.sigmas[clas]))
            probas.append(proba)
        probas=pd.DataFrame(probas,columns=self.classes)
        return probas
    
    def predict(self,X_t):
        probas=np.array(self.predict_proba(X_t))
        ans=[]
        for i in range(probas.shape[0]):
            idx=(np.argmax(probas[i]))
            ans.append(self.classes[idx])
        
        return np.array(ans)