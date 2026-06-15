import numpy as np
import pytest

from mls.LinearModels import (
    _Xy_checker,
    _X_t_checker,
    LinearRegressor,
    SGD_LinearRegressor,
    LogisticRegression
)

# =====================================================
# Input validation
# =====================================================

def test_xy_checker_valid():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])

    X_out, y_out = _Xy_checker(X, y)

    assert X_out.shape == (2, 2)
    assert y_out.shape == (2,)


def test_xy_checker_none():
    with pytest.raises(ValueError):
        _Xy_checker(None, None)


def test_xy_checker_dimension_error():
    X = np.array([1, 2, 3])
    y = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        _Xy_checker(X, y)


def test_xt_checker_feature_mismatch():
    X = np.array([[1, 2]])
    X_t = np.array([[1, 2, 3]])

    with pytest.raises(ValueError):
        _X_t_checker(X_t, X)


# =====================================================
# Linear Regression
# =====================================================

def test_linear_regressor_exact_fit():

    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = 2 * X.flatten() + 1

    model = LinearRegressor()
    model.fit(X, y)

    preds = model.predict(X)

    np.testing.assert_allclose(
        preds,
        y,
        atol=1e-8
    )


def test_linear_regressor_predict_shape():

    X = np.array([[1], [2], [3]])
    y = np.array([3, 5, 7])

    model = LinearRegressor()
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (3,)


# =====================================================
# SGD Linear Regression
# =====================================================

def test_sgd_linear_regressor_learns_line():

    np.random.seed(42)

    X = np.arange(1, 51).reshape(-1, 1)
    y = 3 * X.flatten() + 5

    model = SGD_LinearRegressor(
        alpha=0.001,
        batch_size=10,
        epochs=5000
    )

    model.fit(X, y)

    preds = model.predict(X)

    mse = np.mean((preds - y) ** 2)

    assert mse < 5


def test_sgd_predict_before_fit():

    model = SGD_LinearRegressor()

    with pytest.raises(ValueError):
        model.predict([[1]])


# =====================================================
# Logistic Regression
# =====================================================

def test_logistic_regression_binary_classification():

    np.random.seed(42)

    X = np.array([
        [1],
        [2],
        [3],
        [8],
        [9],
        [10]
    ])

    y = np.array([
        0,
        0,
        0,
        1,
        1,
        1
    ])

    model = LogisticRegression(
        alpha=0.01,
        batch_size=2,
        epochs=5000
    )

    model.fit(X, y)

    preds = model.predict(X)

    accuracy = np.mean(preds == y)

    assert accuracy >= 0.8


def test_logistic_predict_proba_range():

    np.random.seed(42)

    X = np.array([
        [1],
        [2],
        [8],
        [9]
    ])

    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(
        alpha=0.01,
        epochs=3000
    )

    model.fit(X, y)

    probs = model.predict_proba(X)

    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


def test_logistic_more_than_two_classes():

    X = np.array([
        [1],
        [2],
        [3]
    ])

    y = np.array([
        0,
        1,
        2
    ])

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)