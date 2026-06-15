import numpy as np
import pandas as pd
import pytest

from mls.KNeighbours import (
    _Xy_checker,
    _X_t_checker,
    KnnClassifier,
    KnnRegressor
)

# =====================================================
# _Xy_checker
# =====================================================

def test_xy_checker_valid():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    X_out, y_out = _Xy_checker(X, y)

    assert X_out.shape == (2, 2)
    assert y_out.shape == (2,)


def test_xy_checker_none():
    with pytest.raises(ValueError):
        _Xy_checker(None, None)


def test_xy_checker_dimension_error():
    X = np.array([1, 2, 3])
    y = np.array([0, 1, 0])

    with pytest.raises(ValueError):
        _Xy_checker(X, y)


def test_xy_checker_nan_X():
    X = np.array([[1, np.nan]])
    y = np.array([0])

    with pytest.raises(ValueError):
        _Xy_checker(X, y)


def test_xy_checker_nan_y():
    X = np.array([[1, 2]])
    y = np.array([np.nan])

    with pytest.raises(ValueError):
        _Xy_checker(X, y)


# =====================================================
# _X_t_checker
# =====================================================

def test_xt_checker_valid():
    X = np.array([[1, 2], [3, 4]])
    X_t = np.array([[5, 6]])

    result = _X_t_checker(X_t, X)

    assert result.shape == (1, 2)


def test_xt_checker_1d_input():
    X = np.array([[1, 2], [3, 4]])
    X_t = np.array([5, 6])

    result = _X_t_checker(X_t, X)

    assert result.shape == (1, 2)


def test_xt_checker_feature_mismatch():
    X = np.array([[1, 2], [3, 4]])
    X_t = np.array([[1, 2, 3]])

    with pytest.raises(ValueError):
        _X_t_checker(X_t, X)


def test_xt_checker_nan():
    X = np.array([[1, 2]])
    X_t = np.array([[1, np.nan]])

    with pytest.raises(ValueError):
        _X_t_checker(X_t, X)


# =====================================================
# KnnClassifier
# =====================================================

def test_classifier_fit():
    X = np.array([[0], [1], [10], [11]])
    y = np.array([0, 0, 1, 1])

    model = KnnClassifier(k=3)
    model.fit(X, y)

    assert model.X is not None
    assert model.y is not None


def test_classifier_predict_simple():
    X = np.array([[0], [1], [10], [11]])
    y = np.array([0, 0, 1, 1])

    model = KnnClassifier(k=3)
    model.fit(X, y)

    pred = model.predict([[0.5]])

    assert pred[0] == 0


def test_classifier_multiple_predictions():
    X = np.array([[0], [1], [10], [11]])
    y = np.array([0, 0, 1, 1])

    model = KnnClassifier(k=3)
    model.fit(X, y)

    preds = model.predict([[0.5], [10.5]])

    assert preds.shape == (2,)
    assert np.array_equal(preds, np.array([0, 1]))


def test_classifier_default_k():
    model = KnnClassifier()

    assert model.k == 3


# =====================================================
# KnnRegressor
# =====================================================

def test_regressor_fit():
    X = np.array([[1], [2], [3]])
    y = np.array([10, 20, 30])

    model = KnnRegressor(k=2)
    model.fit(X, y)

    assert model.X is not None


def test_regressor_predict():
    X = np.array([[1], [2], [3]])
    y = np.array([10, 20, 30])

    model = KnnRegressor(k=2)
    model.fit(X, y)

    pred = model.predict([[1.5]])

    assert pred.shape == (1,)
    assert pred[0] == pytest.approx(15.0)


def test_regressor_multiple_predictions():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([10, 20, 30, 40])

    model = KnnRegressor(k=2)
    model.fit(X, y)

    preds = model.predict([[1.5], [3.5]])

    assert preds.shape == (2,)


def test_regressor_default_k():
    model = KnnRegressor()

    assert model.k == 3