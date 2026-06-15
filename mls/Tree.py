import numpy as np
import pandas as pd
import heapq

class _TreeNode:
    def __init__(self,y1,y2,threshold,i):
        self.left=None
        self.right=None
        self.threshold=threshold
        self.y1=y1
        self.y2=y2
        self.index=i

class DecisionTreeClassifier:

    def __init__(self,max_depth=float('inf'),min_samples=0,cl=None,max_features=None):
        self.X=None
        self.y=None
        self.root=None
        self.features=None
        self.class_length=None
        self.max_features=max_features
        self.max_depth=max_depth
        self.min_samples=min_samples

    def _X_y_checker(self,X,y):
        if X is None or y is None:
            raise ValueError("X and y can't be None")
        if not(isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)):
            raise ValueError("X has to be a numpy array or DataFrame")
        if not(isinstance(y,pd.Series) or isinstance(y,np.ndarray)):
            raise ValueError("y has to be a numpy array or Series")
        
        classes=None
        X_temp = np.array(X)
        y_temp = np.array(y)
        classes=np.arange(X.shape[1])
        if not np.issubdtype(X_temp.dtype, np.number):
            raise TypeError("X must be numerical or one hot encoded")
        if not np.issubdtype(y_temp.dtype,np.number):
            raise TypeError("y must be numerical")
        
        return X_temp,y_temp,classes

    def _X_t_Checker(self,X_t,X):
        if X_t is None:
            raise ValueError("X_t can't be None")
        if not(isinstance(X_t,pd.DataFrame) or isinstance(X_t,np.ndarray)):
            raise ValueError("X_t has to be a numpy array or DataFrame")
        X_temp = np.array(X_t)
        if not np.issubdtype(X_temp.dtype, np.number):
            raise TypeError("X must be numerical or one hot encoded")
        
        if X_temp.shape[1]!=X.shape[1]:
            raise ValueError("Feature count mismatch between training data and testing data")
        
        return X_temp

    def _selectSplit(self,X,y,features):
        impurities=[]
        unq=self.class_length
        thresholds=np.zeros(X.shape[1])
        for i in features:
            if(len(np.unique(X[:,i])==2)):
                left_mask = X[:, i] == 0
                right_mask = ~left_mask

                left_dist=np.bincount(
                    y[left_mask],
                    minlength= unq
                )
                right_dist=np.bincount(
                    y[right_mask],
                    minlength= unq
                )

                left_sum=np.sum(left_dist)
                right_sum=np.sum(right_dist)
                if left_sum == 0 or right_sum == 0:
                    continue 
                left_dist=np.square(left_dist/left_sum)
                right_dist=np.square(right_dist/right_sum)

                left_imp=1-np.sum(left_dist)
                right_imp=1-np.sum(right_dist)

                imp = (left_imp*left_sum+right_imp*right_sum)/(left_sum+right_sum)
                heapq.heappush(impurities,(imp,i))
            
            else:
                values=np.sort(X[:,i])
                temp=[]
                for j in range(values.shape[0]-1):
                    threshold=(values[j]+values[j+1])/2
                    left_mask=X[:,i]>=threshold
                    right_mask=~left_mask

                    left_dist=np.bincount(
                    y[left_mask],
                    minlength= unq
                    )
                    right_dist=np.bincount(
                        y[right_mask],
                        minlength= unq
                    )

                    left_sum=np.sum(left_dist)
                    right_sum=np.sum(right_dist)
                    if left_sum == 0 or right_sum == 0:
                        continue 
                    left_dist=np.square(left_dist/left_sum)
                    right_dist=np.square(right_dist/right_sum)

                    left_imp=1-np.sum(left_dist)
                    right_imp=1-np.sum(right_dist)

                    imp = (left_imp*left_sum+right_imp*right_sum)/(left_sum+right_sum)
                    heapq.heappush(temp,(imp,(j,threshold)))
                tupl=heapq.heappop(temp)
                thresholds[i]=tupl[1][1]
                heapq.heappush(impurities,(tupl[0],i))
        selectedNode=heapq.heappop(impurities)[1]
        newNode=_TreeNode(y[X[:,selectedNode]>=thresholds[selectedNode]],y[~X[:,selectedNode]>=thresholds[selectedNode]],
                          thresholds[selectedNode],selectedNode)
        return newNode
    
    def _splitValidator(self,parentNode,childNode,left):
        if left:
            temp_y=parentNode.y1
        else:
            temp_y=parentNode.y2
        
        temp_y=np.bincount(temp_y,minlength=self.class_length)
        preSplitImp=1-np.sum(np.square(temp_y/np.sum(temp_y)))
        
        child_y1=np.bincount(childNode.y1,minlength=self.class_length)
        child_y2=np.bincount(childNode.y2,minlength=self.class_length)

        postSplitImp=np.sum(child_y1)*(1-np.sum(np.square(child_y1/np.sum(child_y1))))
        postSplitImp=postSplitImp+np.sum(child_y2)*(1-np.sum(np.square(child_y2/np.sum(child_y2))))
        postSplitImp=postSplitImp/(np.sum(child_y1) + np.sum(child_y2))

        return postSplitImp<preSplitImp
    
    def _buildTree(self,X,y,parentNode: _TreeNode,depth,maxDepth,minSamples):
        if depth>=maxDepth:
            return parentNode
        if y.shape[0]<minSamples:
            return parentNode
        left_mask=X[:,parentNode.index]>=parentNode.threshold
        right_mask=~left_mask

        left_X=X[left_mask]
        left_y=y[left_mask]
        right_X=X[right_mask]
        right_y=y[right_mask]

        if self.max_features is None:
            features=np.arange(self.X.shape[1])
        else:
            features=np.random.choice(self.X.shape[1],self.max_features,replace=False)

        leftNode=self._selectSplit(left_X,left_y,features)
        rightNode=self._selectSplit(right_X,right_y,features)

        if self._splitValidator(parentNode,leftNode,left=True):
            parentNode.left=self._buildTree(left_X,left_y,leftNode,depth+1,
                                            maxDepth,minSamples)
        
        if self._splitValidator(parentNode,rightNode,left=False):
            parentNode.right=self._buildTree(right_X,right_y,rightNode,depth+1,
                                             maxDepth,minSamples)
        
        return parentNode

    def fit(self,X,y,cl):
        self.X,self.y,self.features=self._X_y_checker(X,y)
        self.class_length=len(np.unique(self.y))
        if cl is not None:
            self.class_length=cl

        if self.max_features is not None:
            features=np.arange(self.X.shape[1])
        else:
            features=np.random.choice(self.max_features,self.X.shape[1],replace=False)
        root=self._selectSplit(self.X,self.y,features)
        root=self._buildTree(self.X,self.y,root,[],1,self.max_depth,self.min_samples)
        self.root=root
    
    def _traverseTree(self,tc):
        temp=self.root

        while temp.left is not None and temp.right is not None:
            if tc[np.where(self.features == temp.val)[0][0]]>=temp.threshold:
                if temp.left is None:
                    break
                temp=temp.left
            
            else:
                if temp.right is None:
                    break
                temp=temp.right
        
        return temp

    def predict_proba(self,X_t):
        if self.X is None:
            raise ValueError("Cannot predict without training")
        X_t=self._X_t_Checker(X_t,self.X)
        probas=np.zeros((X_t.shape[0],self.class_length))
        for i in range(X_t.shape[0]):
            ans=self._traverseTree(X_t[i,:])
            if X_t[i,ans.index]>=ans.threshold:
                ans=ans.y1
            else:
                ans=ans.y2
            
            ans=np.bincount(ans,minlength=self.class_length)
            probas[i,:]=ans
        return probas
    
    def predict(self,X_t):
        if self.X is None:
            raise ValueError("Cannot predict without training")
        probas=self.predict_proba(X_t)
        predictions=np.zeros(probas.shape[0],dtype=int)
        for i in range(probas.shape[0]):
            predictions[i]=np.argmax(probas[i,:])
        return predictions
    

class DecisionTreeRegressor:
    def __init__(self,max_depth=float('inf'),min_samples=0,max_features=None):
        self.X=None
        self.y=None
        self.features=None
        self.root=None
        self.max_features=max_features
        self.max_depth=max_depth
        self.min_samples=min_samples
    def _X_y_checker(self,X,y):
        if X is None or y is None:
            raise ValueError("X and y can't be None")
        if not(isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)):
            raise ValueError("X has to be a numpy array or DataFrame")
        if not(isinstance(y,pd.Series) or isinstance(y,np.ndarray)):
            raise ValueError("y has to be a numpy array or Series")
        
        classes=None
        classes=np.arange(X.shape[1])
            
        X_temp = np.array(X)
        y_temp = np.array(y)

        if not np.issubdtype(X_temp.dtype, np.number):
            raise TypeError("X must be numerical or one hot encoded")
        
        if not np.issubdtype(y_temp.dtype,np.number):
            raise TypeError("y must be numerical")
        
        return X_temp,y_temp,classes

    def _X_t_Checker(self,X_t,X):
        if X_t is None:
            raise ValueError("X_t can't be None")
        if not(isinstance(X_t,pd.DataFrame) or isinstance(X_t,np.ndarray)):
            raise ValueError("X_t has to be a numpy array or DataFrame")
        X_temp = np.array(X_t)
        if not np.issubdtype(X_temp.dtype, np.number):
            raise TypeError("X must be numerical or one hot encoded")
        
        if X_temp.shape[1]!=X.shape[1]:
            raise ValueError("Feature count mismatch between training data and testing data")
        
        return X_temp
    
    def _selectSplit(self,X,y,features):
        losses=[]
        thresholds=np.zeros(X.shape[1])
        for i in features:
            if len(np.unique(X[:,i]))==2:
                left_mask= X[:,i]==0
                right_mask=~left_mask

                left_split=y[left_mask]
                right_split=y[right_mask]

                left_pred=np.mean(left_split)
                right_pred=np.mean(right_split)

                curr_loss=np.sum(np.square(left_split-left_pred))+np.sum(np.square(right_split-right_pred))
                heapq.heappush(losses,(curr_loss,i))
            
            else:
                values=np.sort(X[:,i])
                curr_thresholds=[]
                for j in range(values.shape[0]-1):
                    curr_threshold=(values[j]+values[j+1])/2
                    
                    left_mask=X[:,i]>=curr_threshold
                    right_mask=~left_mask

                    left_split=y[left_mask]
                    right_split=y[right_mask]

                    if left_split.size == 0 or right_split.size == 0:
                        continue

                    left_pred=np.mean(left_split)
                    right_pred=np.mean(right_split)

                    curr_loss=np.sum(np.square(left_split-left_pred))+np.sum(np.square(right_split-right_pred))
                    heapq.heappush(curr_thresholds,(curr_loss,curr_threshold))
                if len(curr_thresholds)==0:
                    continue
                best_threshold=heapq.heappop(curr_thresholds)
                thresholds[i]=best_threshold[1]
                heapq.heappush(losses,(best_threshold[0],i))

        selectedSplit=heapq.heappop(losses)[1]
        newNode=_TreeNode(y[X[:,selectedSplit]>=thresholds[selectedSplit]],
                          y[X[:,selectedSplit]<thresholds[selectedSplit]],thresholds[selectedSplit],selectedSplit)
        
        return newNode
    
    def _splitValidator(self,parentNode,childNode,left):
        if left:
            loss=np.sum(np.square(parentNode.y1-np.mean(parentNode.y1)))
        else:
            loss=np.sum(np.square(parentNode.y2-np.mean(parentNode.y2)))
        
        leftLoss=np.sum(np.square(childNode.y1-np.mean(childNode.y1)))
        rightLoss=np.sum(np.square(childNode.y2-np.mean(childNode.y2)))
        childLoss=(childNode.y1.shape[0]*leftLoss+childNode.y2.shape[0]*rightLoss)/(childNode.y1.shape[0]+
                                                                                    childNode.y2.shape[0])
        return childLoss<loss
    
    def _buildTree(self,X,y,parentNode : _TreeNode,depth,maxDepth,min_samples):
        if depth>=maxDepth:
            return parentNode
        if y.shape[0]<min_samples:
            return parentNode
        
        left_y,right_y=parentNode.y1,parentNode.y2
        left_mask=X[:,parentNode.index]>=parentNode.threshold
        right_mask=~left_mask
        left_mask = X[:, parentNode.index] >= parentNode.threshold
        right_mask = ~left_mask

        left_X=X[left_mask,:]
        right_X=X[right_mask,:]

        if self.max_features is None:
            features=np.arange(self.X.shape[1])
        else:
            features=np.random.choice(self.X.shape[1],self.max_features,replace=False)

        leftNode=self._selectSplit(left_X,left_y,features)
        rightNode=self._selectSplit(right_X,right_y,features)

        if self._splitValidator(parentNode,leftNode,True):
            parentNode.left=self._buildTree(left_X,left_y,leftNode,depth+1,maxDepth,min_samples)

        if self._splitValidator(parentNode,rightNode,False):
            parentNode.right=self._buildTree(right_X,right_y,rightNode,depth+1,maxDepth,min_samples)

        return parentNode                    
    
    def fit(self,X,y):
        self.X,self.y,self.features=self._X_y_checker(X,y)
        if self.max_features is None:
            features=np.arange(X.shape[1])
        else:
            features=np.random.choice(self.X.shape[1],self.max_features,replace=False)
        root=self._selectSplit(self.X,self.y,features)
        root=self._buildTree(self.X,self.y,root,[],1,self.max_depth,self.min_samples)
        self.root=root

    def _traverseTree(self,tc):
        temp=self.root
        while temp.left is not None and temp.right is not None:
            if tc[temp.index]>=temp.threshold:
                if temp.left is None:
                    return temp
                temp=temp.left
            else:
                if temp.right is None:
                    return temp
                temp=temp.right
        
        return temp
    def predict(self,X_t):
        if self.X is None:
            raise ValueError("Cannot predict without training")
        X_t=self._X_t_Checker(X_t,self.X)
        ans=np.zeros(X_t.shape[0])
        for i in range(ans.shape[0]):
            leaf=self._traverseTree(X_t[i,:])
            ans[i] = np.mean(leaf.y1) if X_t[i,leaf.index]>=leaf.threshold else np.mean(leaf.y2)
        
        return ans
    
