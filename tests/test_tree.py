import numpy as np

from mls.Tree import DecisionTreeClassifier, DecisionTreeRegressor
from mls.Ensemble import RandomForestClassifier, RandomForestRegressor


def test_decision_tree_classifier():
    X = np.array([
        [0,0],
        [0,0],
        [1,0],
        [1,0],
        [0,1],
        [0,1],
        [1,1],
        [1,1]
    ])

    y = np.array([0,0,1,1,0,0,1,1])

    model = DecisionTreeClassifier(max_depth=10)
    model.fit(X,y)

    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.mean(preds == y) >= 0.5


def test_decision_tree_regressor():
    X = np.array([
        [1],
        [2],
        [3],
        [10],
        [11],
        [12]
    ])

    y = np.array([
        1,
        1,
        1,
        10,
        10,
        10
    ])

    model = DecisionTreeRegressor(max_depth=10)
    model.fit(X,y)

    preds = model.predict(X)

    mse = np.mean((preds - y) ** 2)

    assert preds.shape == y.shape
    assert mse < 1


def test_random_forest_classifier():
    rng = np.random.default_rng(42)

    X = rng.integers(0,2,size=(200,5))
    y = X[:,0]

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        max_features=2
    )

    model.fit(X,y)

    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.mean(preds == y) >= 0.5


def test_random_forest_regressor():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(200,4))
    y = 5 * X[:,0]

    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        max_features=2
    )

    model.fit(X,y)

    preds = model.predict(X)

    mse = np.mean((preds - y) ** 2)

    assert preds.shape == y.shape
    assert mse < 5