import numpy as np
import pandas as pd
import pytest

from mls.Gaussian import (
    _Xy_checker,
    _X_t_checker,
    NaiveBayesClassifier,
    GaussianClassifier
)


# =====================================================
# _Xy_checker
# =====================================================

def test_xy_checker_valid():
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y = pd.Series([0, 1])

    X_out, y_out = _Xy_checker(X, y)

    assert X_out.shape == (2, 2)
    assert y_out.shape == (2,)


def test_xy_checker_none():
    with pytest.raises(ValueError):
        _Xy_checker(None, None)


def test_xy_checker_shape_mismatch():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1])

    with pytest.raises(ValueError):
        _Xy_checker(X, y)


def test_xy_checker_nan_in_X():
    X = pd.DataFrame({"a": [1, np.nan]})
    y = pd.Series([0, 1])

    with pytest.raises(ValueError):
        _Xy_checker(X, y)


def test_xy_checker_nan_in_y():
    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, np.nan])

    with pytest.raises(ValueError):
        _Xy_checker(X, y)


# =====================================================
# _X_t_checker
# =====================================================

def test_xt_checker_valid():
    X = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4]
    })

    X_t = pd.DataFrame({
        "a": [5],
        "b": [6]
    })

    out = _X_t_checker(X_t, X)

    assert out.shape == (1, 2)


def test_xt_checker_missing_column():
    X = pd.DataFrame({
        "a": [1],
        "b": [2]
    })

    X_t = pd.DataFrame({
        "a": [5]
    })

    with pytest.raises(ValueError):
        _X_t_checker(X_t, X)


def test_xt_checker_feature_count_mismatch():
    X = pd.DataFrame({
        "a": [1],
        "b": [2]
    })

    X_t = pd.DataFrame({
        "a": [3],
        "b": [4],
        "c": [5]
    })

    with pytest.raises(ValueError):
        _X_t_checker(X_t, X)


# =====================================================
# NaiveBayesClassifier
# =====================================================

def test_nb_fit():
    X = pd.DataFrame({
        "x1": [1, 2, 8, 9],
        "x2": [1, 1, 8, 8]
    })

    y = pd.Series([0, 0, 1, 1])

    model = NaiveBayesClassifier()
    model.fit(X, y)

    assert model.classes is not None
    assert len(model.classes) == 2


def test_nb_predict_training_data():
    X = pd.DataFrame({
        "x1": [1, 2, 8, 9],
        "x2": [1, 1, 8, 8]
    })

    y = pd.Series([0, 0, 1, 1])

    model = NaiveBayesClassifier()
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (4,)
    assert set(preds).issubset({0, 1})


def test_nb_predict_proba_shape():
    X = pd.DataFrame({
        "x1": [1, 2, 8, 9],
        "x2": [1, 1, 8, 8]
    })

    y = pd.Series([0, 0, 1, 1])

    model = NaiveBayesClassifier()
    model.fit(X, y)

    probs = model.predict_proba(X)

    assert probs.shape == (4, 2)


def test_nb_probability_rows_sum_to_one():
    X = pd.DataFrame({
        "x1": [1, 2, 8, 9],
        "x2": [1, 1, 8, 8]
    })

    y = pd.Series([0, 0, 1, 1])

    model = NaiveBayesClassifier()
    model.fit(X, y)

    probs = model.predict_proba(X)

    np.testing.assert_allclose(
        probs.sum(axis=1),
        np.ones(len(probs)),
        atol=1e-6
    )


def test_nb_predict_before_fit():
    model = NaiveBayesClassifier()

    X = pd.DataFrame({"x": [1]})

    with pytest.raises(ValueError):
        model.predict_proba(X)


# =====================================================
# GaussianClassifier
# =====================================================

def test_gaussian_fit():
    X = pd.DataFrame({
        "x1": [1, 2, 8, 9],
        "x2": [1, 2, 8, 9]
    })

    y = pd.Series([0, 0, 1, 1])

    model = GaussianClassifier()
    model.fit(X, y)

    assert model.means is not None
    assert model.sigmas is not None


def test_gaussian_predict_shape():
    X = pd.DataFrame({
        "x1": [1, 2, 8, 9],
        "x2": [1, 2, 8, 9]
    })

    y = pd.Series([0, 0, 1, 1])

    model = GaussianClassifier()
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (4,)


def test_gaussian_predict_proba_shape():
    X = pd.DataFrame({
        "x1": [1, 2, 8, 9],
        "x2": [1, 2, 8, 9]
    })

    y = pd.Series([0, 0, 1, 1])

    model = GaussianClassifier()
    model.fit(X, y)

    probs = model.predict_proba(X)

    assert probs.shape == (4, 2)


def test_gaussian_predict_before_fit():
    model = GaussianClassifier()

    X = pd.DataFrame({
        "x1": [1],
        "x2": [2]
    })

    with pytest.raises(ValueError):
        model.predict_proba(X)