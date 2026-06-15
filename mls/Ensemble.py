import numpy as np
from .Tree import DecisionTreeClassifier,DecisionTreeRegressor

class RandomForestClassifier:

    def __init__(self,n_estimators=100,max_depth=float('inf'), min_samples=0,max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features

        self.trees = []
        self.class_length = None

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.class_length = len(np.unique(y))
        self.trees = []

        n_samples = X.shape[0]

        for _ in range(self.n_estimators):

            indices = np.random.choice(
                n_samples,
                n_samples,
                replace=True
            )

            X_boot = X[indices]
            y_boot = y[indices]

            tree = DecisionTreeClassifier(max_depth=self.max_depth,min_samples=self.min_samples,max_features=self.max_features)

            tree.fit(
                X_boot,
                y_boot,
                cl=self.class_length,
            )

            self.trees.append(tree)

    def predict_proba(self, X):

        X = np.asarray(X)

        probas = np.zeros(
            (X.shape[0], self.class_length),
            dtype=float
        )

        for tree in self.trees:
            probas += tree.predict_proba(X)

        probas /= len(self.trees)

        row_sums = probas.sum(axis=1, keepdims=True)
        probas /= row_sums

        return probas

    def predict(self, X):

        probas = self.predict_proba(X)

        return np.argmax(probas, axis=1)
    
class RandomForestRegressor:

    def __init__(self,n_estimators=100,max_depth=float('inf'),min_samples=0,max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features

        self.trees = []

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.trees = []

        n_samples = X.shape[0]

        for _ in range(self.n_estimators):

            indices = np.random.choice(
                n_samples,
                n_samples,
                replace=True
            )

            X_boot = X[indices]
            y_boot = y[indices]

            tree = DecisionTreeRegressor(max_depth=self.max_depth,min_samples=self.min_samples,max_features=self.max_features)

            tree.fit(X_boot,y_boot)

            self.trees.append(tree)

    def predict(self, X_t):
        if self.trees is None:
            raise ValueError("Cannot predict without fitting")
        X_t = np.asarray(X_t)

        predictions = np.zeros(
            (len(self.trees), X_t.shape[0]),
            dtype=float
        )

        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(X_t)

        return np.mean(predictions, axis=0)