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

    def __init__(self,max_depth=10e8,min_samples=0,cl=None,max_features=None):
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
                if len(temp)==0:
                    continue
                tupl=heapq.heappop(temp)
                thresholds[i]=tupl[1][1]
                heapq.heappush(impurities,(tupl[0],i))
        if len(impurities)==0:
            return None
        selectedNode=heapq.heappop(impurities)[1]
        newNode=_TreeNode(y[X[:,selectedNode]>=thresholds[selectedNode]],y[X[:,selectedNode]<thresholds[selectedNode]],
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
            features = np.arange(self.X.shape[1])
        else:
            features = np.random.choice(
                self.X.shape[1],
                self.max_features,
                replace=False
            )

        leftNode=self._selectSplit(left_X,left_y,features)
        rightNode=self._selectSplit(right_X,right_y,features)

        if leftNode is not None and self._splitValidator(parentNode,leftNode,left=True):
            parentNode.left=self._buildTree(left_X,left_y,leftNode,depth+1,
                                            maxDepth,minSamples)
        
        if rightNode is not None and self._splitValidator(parentNode,rightNode,left=False):
            parentNode.right=self._buildTree(right_X,right_y,rightNode,depth+1,
                                             maxDepth,minSamples)
        
        return parentNode

    def fit(self,X,y,cl=None):
        self.X,self.y,self.features=self._X_y_checker(X,y)
        self.class_length=len(np.unique(self.y))
        if cl is not None:
            self.class_length=cl

        if self.max_features is None:
            features = np.arange(self.X.shape[1])
        else:
            features = np.random.choice(
                self.X.shape[1],
                self.max_features,
                replace=False
            )
        root=self._selectSplit(self.X,self.y,features)
        if root is not None:
            root=self._buildTree(self.X,self.y,root,1,self.max_depth,self.min_samples)
        self.root=root
    
    def _traverseTree(self,tc):
        temp=self.root
        if temp is None:
            raise ValueError("Absolutely nothing was able to be learnt from the data")
        while temp.left is not None and temp.right is not None:
            if tc[temp.index]>=temp.threshold:
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

    def __init__(self, max_depth=10, min_samples=2, max_features=None):
        self.root = None
        self.X = None
        self.y = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples
        self.max_features = max_features

    def _check(self, X, y):
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        return X, y

    def _mse(self, y):
        if y.size == 0:
            return 0.0
        return np.mean((y - np.mean(y)) ** 2)

    def _selectSplit(self, X, y, features):
        best_loss = float("inf")
        best_node = None

        for i in features:

            values = np.unique(X[:, i])
            if len(values) == 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2

            for t in thresholds:
                left_mask = X[:, i] <= t
                right_mask = ~left_mask

                y1 = y[left_mask]
                y2 = y[right_mask]

                if y1.size == 0 or y2.size == 0:
                    continue

                loss = (
                    y1.size * self._mse(y1) +
                    y2.size * self._mse(y2)
                ) / y.size

                if loss < best_loss:
                    best_loss = loss
                    best_node = _TreeNode(y1, y2, t, i)

        return best_node

    def _buildTree(self, X, y, node, depth):

        if node is None:
            return None

        if (
            depth >= self.max_depth
            or y.size < self.min_samples_split
            or np.all(y == y[0])
        ):
            return node

        left_mask = X[:, node.index] <= node.threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return node

        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        if self.max_features is None:
            features = np.arange(X.shape[1])
        else:
            features = np.random.choice(
                X.shape[1],
                self.max_features,
                replace=False
            )

        node.left = self._selectSplit(left_X, left_y, features)
        node.right = self._selectSplit(right_X, right_y, features)

        node.left = self._buildTree(left_X, left_y, node.left, depth + 1)
        node.right = self._buildTree(right_X, right_y, node.right, depth + 1)

        return node

    def fit(self, X, y):
        self.X, self.y = self._check(X, y)

        features = np.arange(self.X.shape[1])

        self.root = self._selectSplit(self.X, self.y, features)

        self.root = self._buildTree(self.X, self.y, self.root, depth=1)

    def _traverse(self, x):
        node = self.root

        if node is None:
            raise ValueError("Model not trained")

        while node.left is not None or node.right is not None:
            if x[node.index] <= node.threshold:
                if node.left is None:
                    break
                node = node.left
            else:
                if node.right is None:
                    break
                node = node.right

        return node

    def predict(self, X):
        if self.root is None:
            raise ValueError("Model not trained")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.array(X, dtype=float)

        preds = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            leaf = self._traverse(X[i])

            # SAFE prediction (prevents NaN)
            y_all = np.concatenate([leaf.y1, leaf.y2])

            if y_all.size == 0:
                preds[i] = 0.0
            else:
                preds[i] = np.mean(y_all)

        return preds