import numpy as np
import pytest

from mls.Tree import DecisionTreeClassifier,DecisionTreeRegressor


def test_classifier_fit_predict_simple():

    X = np.array([
        [0],
        [0],
        [1],
        [1]
    ])

    y = np.array([
        0,
        0,
        1,
        1
    ])

    clf = DecisionTreeClassifier()

    clf.fit(X, y, cl=2)

    preds = clf.predict(X)

    assert np.array_equal(preds, y)


def test_classifier_predict_before_fit():

    clf = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        clf.predict(np.array([[0]]))


def test_classifier_predict_proba_shape():

    X = np.array([
        [0],
        [1]
    ])

    y = np.array([
        0,
        1
    ])

    clf = DecisionTreeClassifier()

    clf.fit(X, y, cl=2)

    probs = clf.predict_proba(X)

    assert probs.shape == (2, 2)


def test_classifier_x_feature_mismatch():

    X = np.array([
        [0],
        [1]
    ])

    y = np.array([
        0,
        1
    ])

    clf = DecisionTreeClassifier()

    clf.fit(X, y, cl=2)

    with pytest.raises(ValueError):
        clf.predict(np.array([[0, 1]]))

def test_regressor_fit_predict_simple():

    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([
        1,
        2,
        3,
        4
    ])

    reg = DecisionTreeRegressor()

    reg.fit(X, y)

    preds = reg.predict(X)

    assert preds.shape == y.shape


def test_regressor_predict_before_fit():

    reg = DecisionTreeRegressor()

    with pytest.raises(ValueError):
        reg.predict(np.array([[1]]))


def test_regressor_feature_mismatch():

    X = np.array([
        [1],
        [2]
    ])

    y = np.array([
        1,
        2
    ])

    reg = DecisionTreeRegressor()

    reg.fit(X, y)

    with pytest.raises(ValueError):
        reg.predict(np.array([[1, 2]]))


def test_classifier_max_features():

    X = np.random.rand(100, 5)

    y = (X[:, 0] > 0.5).astype(int)

    clf = DecisionTreeClassifier(
        max_features=3
    )

    clf.fit(X, y, cl=2)

    preds = clf.predict(X)

    assert preds.shape == y.shape


def test_regressor_max_features():

    X = np.random.rand(100, 5)

    y = X[:, 0] + X[:, 1]

    reg = DecisionTreeRegressor(
        max_features=3
    )

    reg.fit(X, y)

    preds = reg.predict(X)

    assert preds.shape == y.shape

def test_classifier_training_accuracy():

    np.random.seed(42)

    X = np.random.rand(200, 4)

    y = (X[:, 0] > 0.5).astype(int)

    clf = DecisionTreeClassifier()

    clf.fit(X, y, cl=2)

    preds = clf.predict(X)

    acc = np.mean(preds == y)

    assert acc > 0.95