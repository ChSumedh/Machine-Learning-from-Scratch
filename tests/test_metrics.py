import numpy as np
import pandas as pd
import pytest

from mls.Metrics import (
    _Y_checker,
    confusion_matrix,
    classification_report,
    accuracy_score,
    rmse
)

# =====================================================
# _Y_checker
# =====================================================

def test_y_checker_valid():
    y = np.array([0, 1, 0])
    y_t = np.array([0, 1, 1])

    _Y_checker(y, y_t)


def test_y_checker_none():
    with pytest.raises(ValueError):
        _Y_checker(None, np.array([1]))


def test_y_checker_shape_mismatch():
    y = np.array([0, 1])
    y_t = np.array([0])

    with pytest.raises(ValueError):
        _Y_checker(y, y_t)


def test_y_checker_nan():
    y = np.array([0, 1])
    y_t = np.array([0, np.nan])

    with pytest.raises(ValueError):
        _Y_checker(y, y_t)


# =====================================================
# confusion_matrix
# =====================================================

def test_confusion_matrix_binary():

    y = np.array([0, 0, 1, 1])
    y_t = np.array([0, 1, 0, 1])

    cm = confusion_matrix(y, y_t)

    assert cm.loc[0, 0] == 1
    assert cm.loc[0, 1] == 1
    assert cm.loc[1, 0] == 1
    assert cm.loc[1, 1] == 1


def test_confusion_matrix_multiclass():

    y = np.array([0, 1, 2])
    y_t = np.array([0, 2, 1])

    cm = confusion_matrix(y, y_t)

    assert cm.shape == (3, 3)
    assert cm.loc[0, 0] == 1
    assert cm.loc[1, 2] == 1
    assert cm.loc[2, 1] == 1


# =====================================================
# accuracy_score
# =====================================================

def test_accuracy_score_perfect():

    y = np.array([0, 1, 2])
    y_t = np.array([0, 1, 2])

    assert accuracy_score(y, y_t) == 1.0


def test_accuracy_score_half():

    y = np.array([0, 0, 1, 1])
    y_t = np.array([0, 1, 1, 0])

    assert accuracy_score(y, y_t) == 0.5


# =====================================================
# classification_report
# =====================================================

def test_classification_report_columns():

    y = np.array([0, 0, 1, 1])
    y_t = np.array([0, 1, 0, 1])

    report = classification_report(y, y_t)

    assert list(report.columns) == [
        "Precision",
        "Recall",
        "F1-Score"
    ]


def test_classification_report_perfect():

    y = np.array([0, 1, 0, 1])
    y_t = np.array([0, 1, 0, 1])

    report = classification_report(y, y_t)

    np.testing.assert_allclose(
        report["Precision"].values,
        np.ones(2)
    )

    np.testing.assert_allclose(
        report["Recall"].values,
        np.ones(2)
    )

    np.testing.assert_allclose(
        report["F1-Score"].values,
        np.ones(2)
    )


# =====================================================
# RMSE
# =====================================================

def test_rmse_zero():

    y = np.array([1, 2, 3])
    y_t = np.array([1, 2, 3])

    assert rmse(y, y_t) == 0


def test_rmse_known_value():

    y = np.array([1, 2, 3])
    y_t = np.array([2, 3, 4])

    expected = np.sqrt((1 + 1 + 1) / 3)

    assert np.isclose(
        rmse(y, y_t),
        expected
    )


def test_rmse_shape_mismatch():

    y = np.array([1, 2])
    y_t = np.array([1])

    with pytest.raises(ValueError):
        rmse(y, y_t)