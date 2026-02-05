import numpy as np
import pandas as pd
class StandardScaler:

    def __init__(self):
        self.means=None
        self.std=None

    @staticmethod
    def X_checker(X):
        if X is None:
            raise ValueError("Inputs can't be none")
        
        if not(isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)):
            raise ValueError("X has to be a numpy array or DataFrame")
        
        if X.ndim!=2:
            raise ValueError("X has to be 2 dimensional")

        temp=pd.DataFrame(X)
        if temp.isna().sum().sum():
            raise ValueError("There shouldn't be NaN values in X")
        return temp

    def fit(self,X):
        self.X=self.X_checker(X)
        self.cols=list(self.X.columns)
        self.X_=self.X.select_dtypes(include=np.number)
        self.means={col:self.X_[col].mean() for col in self.X_.columns}
        self.std={col:self.X_[col].std() for col in self.X_.columns}
        return self
    
    def transform(self,T):
        t=self.X_checker(T).copy()
        t_cols=list(t.columns)
        if self.means is None or self.std is None:
            raise ValueError("Cannot transform without fitting")
        
        for col in t_cols:
            if col not in self.cols:
                raise ValueError("The columns of transformed data not present in the data you fit")
            
            if t[col].dtype!=self.X[col].dtype:
                raise ValueError(f"dtype mismatch between in column {col}")
            
            if pd.api.types.is_numeric_dtype(t[col]):
                t[col]=(t[col]-self.means[col])/self.std[col]
        return t
    
    def fit_transform(self,X):
        self.fit(X)
        t=self.transform(X)
        return t
    
    def get_params(self):
        if self.means is None or self.std is None:
            raise ValueError("Cannot give the params without fitting")
        params=pd.DataFrame([],index=self.cols,columns=['Mean','Standard Deviation'])
        for col in self.cols:
            params.loc[col,'Mean']=self.means[col]
            params.loc[col,'Standard Deviation']=self.std[col]
        
        return params

class MinMaxScaler:

    def __init__(self):
        self.means=None
        self.range=None

    @staticmethod
    def X_checker(X):
        if X is None:
            raise ValueError("Inputs can't be none")
        
        if not(isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)):
            raise ValueError("X has to be a numpy array or DataFrame")
        
        if X.ndim!=2:
            raise ValueError("X has to be 2 dimensional")

        temp=pd.DataFrame(X)
        if temp.isna().sum().sum():
            raise ValueError("There shouldn't be NaN values in X")
        return temp

    def fit(self,X):
        self.X=self.X_checker(X)
        self.cols=list(self.X.columns)
        self.X_=self.X.select_dtypes(include=np.number)
        self.means={col:self.X_[col].mean() for col in self.X_.columns}
        self.range={col:self.X_[col].max()-self.X_[col].min() for col in self.X_.columns}
        return self
    
    def transform(self,T):
        t=self.X_checker(T).copy()
        t_cols=list(t.columns)
        if self.means is None or self.range is None:
            raise ValueError("Cannot transform without fitting")
        
        for col in t_cols:
            if col not in self.cols:
                raise ValueError("The columns of transformed data not present in the data you fit")
            
            if t[col].dtype!=self.X[col].dtype:
                raise ValueError(f"dtype mismatch in column {col}")
            
            if pd.api.types.is_numeric_dtype(t[col]):
                t[col]=(t[col]-self.means[col])/self.range[col]
        return t
    
    def fit_transform(self,X):
        self.fit(X)
        t=self.transform(X)
        return t
    
    def get_params(self):
        if self.means is None or self.range is None:
            raise ValueError("Cannot give the params without fitting")
        params=pd.DataFrame([],index=self.cols,columns=['Mean','Standard Deviation'])
        for col in self.cols:
            params.loc[col,'Mean']=self.means[col]
            params.loc[col,'Range']=self.range[col]
        
        return params
    
class LabelEncoder:
    def __init__(self):
        self.X=None
        self.labels=None
        self.enums=None
        self.enums_inv=None
    
    @staticmethod
    def X_checker(X):
        if X is None:
            raise ValueError("Inputs can't be none")
        
        if not(isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)):
            raise ValueError("X has to be a numpy array or DataFrame")
        
        if X.ndim!=2:
            raise ValueError("X has to be 2 dimensional")

        temp=pd.DataFrame(X)
        if temp.isna().sum().sum():
            raise ValueError("There shouldn't be NaN values in X")
        return temp
    
    def fit(self,X):
        self.X=self.X_checker(X).copy()
        self.labels = {col:sorted(self.X[col].unique()) for col in self.X.columns if 
                       not pd.api.types.is_numeric_dtype(self.X[col])}
        self.enums = {col:{cat: idx for idx, cat in enumerate(self.labels[col])} for col in self.X.columns if
                      not pd.api.types.is_numeric_dtype(self.X[col])}
        self.enums_inv={col:{idx:cat for idx,cat in enumerate(self.labels[col])} for col in self.X.columns if
                        not pd.api.types.is_numeric_dtype(self.X[col])}
        return self
    
    def transform(self,T):
        if not self.enums:
            raise ValueError("Cannot transform without no prior fitted data")
        t=self.X_checker(T).copy()
        for col in t.columns:
            if col in self.enums.keys():
            
                if t[col].dtype != self.X[col].dtype:
                    raise ValueError(f"dtype mismatch betwwen the fitted data and data to be transformed in {col}")
                
                if col in self.labels:
                    unknown_cats = set(t[col].unique()) - set(self.labels[col])
                    if unknown_cats:
                        raise ValueError(f"Unknown categories in {col}: {unknown_cats}")
                
                if not pd.api.types.is_numeric_dtype(t[col]):
                    t[col]=t[col].map(self.enums[col])
        return t
    
    def inverse_transform(self,T):
        if not self.enums_inv:
            raise ValueError("Cannot transform without no prior fitted data")
        t=self.X_checker(T).copy()
        for col in t.columns:
            if col in self.enums_inv.keys():
                if pd.api.types.is_numeric_dtype(t[col]) and not  pd.api.types.is_numeric_dtype(self.X[col]):
                    unknown_codes = set(t[col].unique()) - set(self.enums_inv[col].keys())
                    if unknown_codes:
                        raise ValueError(f"Unknown encoded values in {col}: {unknown_codes}")
                t[col]=t[col].map(self.enums_inv[col])
        return t
        
    
    def fit_transform(self,X):
        self.fit(X)
        t=self.transform(X)
        return t
    
    def get_params(self):
        if not self.enums:
            raise ValueError("Cannot give the params without fitting")
        dfs = []
        for col, mapping in self.enums.items():
            temp_df = pd.DataFrame(list(mapping.items()), columns=['Category', 'Code'])
            temp_df['Column'] = col 
            dfs.append(temp_df)

        df = pd.concat(dfs, ignore_index=True)
        return df
    
class OrdinalEncoder:
    def __init__(self,unknown_map=None):
        self.X=None
        self.map={}
        self.inv_map={}
        self.unknown_map=unknown_map

    @staticmethod
    def X_checker(X):
        if X is None:
            raise ValueError("Inputs can't be none")
        
        if not(isinstance(X,pd.DataFrame)):
            raise ValueError("X has to be a DataFrame")

        temp=pd.DataFrame(X,columns=X.columns)
        if temp.isna().sum().sum():
            raise ValueError("There shouldn't be NaN values in X")
        return temp
    
    def fit(self,X,order=None):
        self.X=self.X_checker(X).copy()
        if order is None:
            raise ValueError("Order of classes for categorical columns has to be given" \
            "in the form { categorical column name: [list of categories in it]}")
        for key,value in order.items():
            if key not in list(self.X.columns):
                raise ValueError(f"{key} does not exist in the data you fit")
            if pd.api.types.is_numeric_dtype(self.X[key]):
                raise ValueError(f"Dtype mismatch for column {key} ")
            if sorted(list(value))!=sorted(list(self.X[key].unique())):
                raise ValueError(f"The categories you entered for {key} don't exist in the data fit")
            
            self.map[key]={cat:idx for idx,cat in enumerate(list(value))}
            self.inv_map[key]={idx:cat for idx,cat in enumerate(list(value))}
        
        return self
    
    def transform(self,T):
        t=self.X_checker(T).copy()
        if not self.map:
            raise ValueError("Cannot transform without fitting data")
        for col in t.columns:
            if col in self.map.keys():
                if t[col].dtype!=self.X[col].dtype:
                    raise ValueError(f"Dtype mismatch between fit data and transform data in the column {col}")
                
                if self.unknown_map is None:
                    unknown_vals = set(t[col].unique()) - set(self.map[col].keys())
                    if unknown_vals:
                        raise ValueError(f"Unknown categories: {unknown_vals}")
                    
                t[col] = t[col].apply(
                        lambda x: self.map[col].get(x, self.unknown_map)
                    )
        
        return t
    
    def inverse_transform(self,T):
        t=self.X_checker(T).copy()
        if not self.inv_map:
            raise ValueError("Cannot transform without fitting data")
        
        for col in t.columns:
            if col in self.inv_map.keys():
                if pd.api.types.is_numeric_dtype(t[col]) and not  pd.api.types.is_numeric_dtype(self.X[col]):
                    unknown_codes = set(t[col].unique()) - set(self.inv_map[col].keys())
                    if unknown_codes:
                        raise ValueError(f"Unknown encoded values in {col}: {unknown_codes}")
                t[col]=t[col].map(self.inv_map[col])
        
        return t

    def fit_transform(self,X,order=None):
        self.fit(X,order=order)
        t=self.transform(X)
        return t
    
    def get_params(self):
        if not self.map:
            raise ValueError("Cannot give the params without fitting")
        dfs = []
        for col, mapping in self.map.items():
            temp_df = pd.DataFrame(list(mapping.items()), columns=['Category', 'Code'])
            temp_df['Column'] = col 
            dfs.append(temp_df)

        df = pd.concat(dfs, ignore_index=True)
        return df

class OneHotEncoder:
    def __init__(self,handle_unknown='halt'):
        self.X=None
        if handle_unknown not in ['ignore','bucket','halt']:
            raise ValueError(f"handle_unknown has to be either one of 'ignore' 'bucket' 'halt'")
        self.handle_unknown=handle_unknown
        self.encoded={}
        self.columns=None
        self.encoded_cols=[]
    
    @staticmethod
    def X_checker(X):
        if X is None:
            raise ValueError("Inputs can't be none")
        
        if not(isinstance(X,pd.DataFrame)):
            raise ValueError("X has to be a DataFrame")

        temp=pd.DataFrame(X,columns=X.columns)
        if temp.isna().sum().sum():
            raise ValueError("There shouldn't be NaN values in X")
        return temp
    
    def fit(self,X,columns=None):
        self.X=self.X_checker(X).copy()
        self.columns=columns
        for col in self.X.columns:
            if self.columns is not None:
                if col in self.columns and not pd.api.types.is_numeric_dtype(self.X[col]):
                    self.encoded[col]=list(self.X[col].unique())
                    self.encoded_cols.append(col)
            else:
                if not pd.api.types.is_numeric_dtype(self.X[col]):
                    self.encoded[col]=list(self.X[col].unique())
                    self.encoded_cols.append(col)
        return self
    
    def __encoder(self,t,cats,col):
        for cat in cats:
            if cat not in self.encoded[col]:
                if self.handle_unknown=='halt':
                    raise ValueError(f"Unknown categories encountered {cat}")
                elif self.handle_unknown=='bucket':
                    t['unknown']=(t[col]==cat).astype(int)
                else:
                    t[f'{col}_{cat}']=(t[col]==cat).astype(int)
            else:
                t[f'{col}_{cat}']=(t[col]==cat).astype(int)
        return t
    
    def transform(self,T):
        if not self.encoded:
            raise ValueError("Cannot transform with no fit data")
        t=self.X_checker(T).copy()
        for col in t.columns:
            if col in self.encoded_cols:
                if t[col].dtype!=self.X[col].dtype:
                    raise ValueError(f"dtype mismatch between fit data and transform data in column {col}")
                
                t=self.__encoder(t,list(t[col].unique()),col)
        return t
    
    def fit_transform(self,X,columns=None):
        self.fit(X,columns=columns)
        t=self.transform(X)
        return t
    
    def get_params(self):
        x=pd.DataFrame(self.encoded)
        encoded=pd.DataFrame(np.reshape(np.array(x),(1,-1)),index=self.encoded.keys())
        return (self.encoded_cols,encoded)

class SimpleImputer:

    def __init__(self,strat='mean',fixed_val=None):
        self.X=None
        if strat not in ['mean','median','mode','fixed']:
            raise ValueError(f"strat has to be one of {['mean','median','mode','fixed']}")
        self.strat=strat
        if self.strat=='fixed':
            if not isinstance(fixed_val,dict):
                raise ValueError("fixed value has to be a dictionary in the form 'column':value")
            self.fixed_val=fixed_val
        self.fill_vals={}
    
    @staticmethod
    def X_checker(X):
        if X is None:
            raise ValueError("Inputs can't be none")
        
        if not(isinstance(X,pd.DataFrame)):
            raise ValueError("X has to be a DataFrame")

        temp=pd.DataFrame(X,columns=X.columns)
        return temp
    
    def fit(self,X):
        self.X=self.X_checker(X)
        for col in self.X.columns:
            if self.X[col].isna().sum():
                if pd.api.types.is_numeric_dtype(self.X[col]):
                    if self.strat=='mean':
                        self.fill_vals[col]=self.X[col].mean()
                    elif self.strat=='median':
                        self.fill_vals[col]=self.X[col].median()
                    elif self.strat=='mode':
                        self.fill_vals[col]=self.X[col].mode()[0]
                    else:
                        self.fill_vals[col]=self.fixed_val[col]
                else: 
                    if self.strat=='mode':
                        self.fill_vals[col]=self.X[col].mode()[0]
                    elif self.strat=='fixed':
                        if col not in self.fixed_val:
                            raise ValueError(f"Column '{col}' not in fixed_val")
                        self.fill_vals[col]=self.fixed_val[col]
        return self
    
    def transform(self,T):
        t=self.X_checker(T)
        for col in t.columns:
            if col in list(self.X.columns):
                if t[col].dtype!=self.X[col].dtype:
                    raise ValueError(f"dtype mismatch between fit data and transform data in column {col}")
                t[col]=t[col].fillna(self.fill_vals[col])
        
        return t
    
    def fit_transform(self,X):
        self.fit(X)
        t=self.transform(X)
        return t
    
    def get_params(self):
        params = pd.DataFrame([self.fill_vals]).T
        params.columns = ['Fill Value'] 
        return params