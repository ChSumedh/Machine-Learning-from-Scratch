import numpy as np
import pandas as pd
import pytest

from mls.ModelSelection import _Xy_checker, split

# =====================================================
# _Xy_checker
# =====================================================

def test_xy_checker_valid_numpy():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])

    X_out, y_out = _Xy_checker(X, y)

    assert X_out.shape == (3, 1)
    assert y_out.shape == (3,)


def test_xy_checker_shape_mismatch():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        _Xy_checker(X, y)


def test_xy_checker_nan():
    X = np.array([[1], [np.nan]])
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        _Xy_checker(X, y)


# =====================================================
# split - numpy
# =====================================================

def test_split_numpy_sizes():

    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)

    X_train, y_train, X_test, y_test = split(
        X,
        y,
        split_size=0.8,
        stratify_y=False,
        random_state=42
    )

    assert len(X_train) == 40
    assert len(X_test) == 10
    assert len(y_train) == 40
    assert len(y_test) == 10


def test_split_reproducible():

    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)

    out1 = split(X, y, 0.8, False, random_state=42)
    out2 = split(X, y, 0.8, False, random_state=42)

    for a, b in zip(out1, out2):
        np.testing.assert_array_equal(a, b)


# =====================================================
# split - pandas
# =====================================================

def test_split_pandas():

    X = pd.DataFrame({
        "a": range(20),
        "b": range(20, 40)
    })

    y = pd.Series(range(20))

    X_train, y_train, X_test, y_test = split(
        X,
        y,
        0.75,
        False,
        random_state=1
    )

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)

    assert len(X_train) == 15
    assert len(X_test) == 5


# =====================================================
# stratification
# =====================================================

def test_stratified_split_preserves_ratio():

    X = np.arange(200).reshape(100, 2)

    y = np.array(
        [0] * 80 +
        [1] * 20
    )

    X_train, y_train, X_test, y_test = split(
        X,
        y,
        split_size=0.8,
        stratify_y=True,
        random_state=42
    )

    train_ratio = np.mean(y_train == 1)
    test_ratio = np.mean(y_test == 1)

    assert abs(train_ratio - 0.20) < 0.05
    assert abs(test_ratio - 0.20) < 0.05


def test_stratified_split_sizes():

    X = np.arange(40).reshape(20, 2)

    y = np.array(
        [0] * 10 +
        [1] * 10
    )

    X_train, y_train, X_test, y_test = split(
        X,
        y,
        split_size=0.7,
        stratify_y=True,
        random_state=0
    )

    assert len(X_train) + len(X_test) == 20
    assert len(y_train) + len(y_test) == 20


# =====================================================
# invalid inputs
# =====================================================

def test_stratify_not_bool():

    X = np.array([[1], [2]])
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        split(X, y, 0.8, "yes")


def test_invalid_X():

    y = np.array([0, 1])

    with pytest.raises(ValueError):
        split(None, y, 0.8, False)