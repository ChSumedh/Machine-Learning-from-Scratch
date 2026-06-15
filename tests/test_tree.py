import numpy as np
import pytest

from mls.Tree import DecisionTreeClassifier, DecisionTreeRegressor
from mls.Ensemble import RandomForestClassifier, RandomForestRegressor


# ----------------------------
# DecisionTreeClassifier
# ----------------------------

def test_classifier_fit_predict():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0],
        [1, 1]
    ])

    y = np.array([0, 0, 1, 1, 0, 1])

    model = DecisionTreeClassifier()
    model.fit(X, y, max_depth=5)

    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.mean(preds == y) >= 0.9


def test_classifier_predict_proba_shape():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier()
    model.fit(X, y)

    probas = model.predict_proba(X)

    assert probas.shape == (4, 2)


def test_classifier_max_features():
    rng = np.random.default_rng(42)

    X = rng.integers(0, 2, size=(100, 5))
    y = X[:, 0]

    model = DecisionTreeClassifier()
    model.fit(X, y, max_depth=10, max_features=2)

    preds = model.predict(X)

    assert np.mean(preds == y) >= 0.8


# ----------------------------
# DecisionTreeRegressor
# ----------------------------

def test_regressor_fit_predict():
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

    model = DecisionTreeRegressor()
    model.fit(X, y, max_depth=5)

    preds = model.predict(X)

    mse = np.mean((preds - y) ** 2)

    assert preds.shape == y.shape
    assert mse < 1.0


def test_regressor_max_features():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(200, 4))
    y = 5 * X[:, 0]

    model = DecisionTreeRegressor()
    model.fit(X, y, max_depth=10, max_features=2)

    preds = model.predict(X)

    mse = np.mean((preds - y) ** 2)

    assert mse < 5


# ----------------------------
# RandomForestClassifier
# ----------------------------

def test_rf_classifier_fit_predict():
    rng = np.random.default_rng(42)

    X = rng.integers(0, 2, size=(200, 4))
    y = (X[:, 0] ^ X[:, 1]).astype(int)

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        max_features=2
    )

    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.mean(preds == y) > 0.9


def test_rf_classifier_predict_proba():
    rng = np.random.default_rng(42)

    X = rng.integers(0, 2, size=(100, 3))
    y = X[:, 0]

    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        max_features=2
    )

    model.fit(X, y)

    probas = model.predict_proba(X)

    assert probas.shape == (100, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


# ----------------------------
# RandomForestRegressor
# ----------------------------

def test_rf_regressor_fit_predict():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(200, 3))
    y = 3 * X[:, 0] - 2 * X[:, 1]

    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        max_features=2
    )

    model.fit(X, y)

    preds = model.predict(X)

    mse = np.mean((preds - y) ** 2)

    assert preds.shape == y.shape
    assert mse < 2


# ----------------------------
# Error handling
# ----------------------------

def test_classifier_predict_before_fit():
    model = DecisionTreeClassifier()

    with pytest.raises(Exception):
        model.predict(np.array([[0, 0]]))


def test_regressor_predict_before_fit():
    model = DecisionTreeRegressor()

    with pytest.raises(Exception):
        model.predict(np.array([[1]]))


def test_rf_classifier_predict_before_fit():
    model = RandomForestClassifier()

    with pytest.raises(Exception):
        model.predict(np.array([[0, 0]]))


def test_rf_regressor_predict_before_fit():
    model = RandomForestRegressor()

    with pytest.raises(Exception):
        model.predict(np.array([[1]]))