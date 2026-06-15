import numpy as np
import pandas as pd
import pytest

from mls.PreProcessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer
)

# =====================================================
# StandardScaler
# =====================================================

def test_standard_scaler_fit_transform():

    X = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50]
    })

    scaler = StandardScaler()
    scaler.fit(X)

    Xt = scaler.transform(X)

    assert np.allclose(Xt["A"].mean(), 0, atol=1e-7)
    assert np.allclose(Xt["B"].mean(), 0, atol=1e-7)


def test_standard_scaler_shape():

    X = pd.DataFrame({
        "A": [1, 2, 3]
    })

    scaler = StandardScaler()
    scaler.fit(X)

    Xt = scaler.transform(X)

    assert Xt.shape == X.shape


# =====================================================
# MinMaxScaler
# =====================================================

def test_minmax_scaler_range():

    X = pd.DataFrame({
        "A": [0, 5, 10]
    })

    scaler = MinMaxScaler()
    scaler.fit(X)

    Xt = scaler.transform(X)

    assert np.isclose(Xt["A"].min(), 0)
    assert np.isclose(Xt["A"].max(), 1)


def test_minmax_scaler_shape():

    X = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

    scaler = MinMaxScaler()
    scaler.fit(X)

    Xt = scaler.transform(X)

    assert Xt.shape == X.shape


# =====================================================
# OneHotEncoder
# =====================================================

def test_onehot_encoder_creates_columns():

    X = pd.DataFrame({
        "Color": ["Red", "Blue", "Red"]
    })

    enc = OneHotEncoder()
    enc.fit(X)

    Xt = enc.transform(X)

    assert "Color_Red" in Xt.columns
    assert "Color_Blue" in Xt.columns


def test_onehot_encoder_values():

    X = pd.DataFrame({
        "Color": ["Red", "Blue"]
    })

    enc = OneHotEncoder()
    enc.fit(X)

    Xt = enc.transform(X)

    red_row = Xt.iloc[0]

    assert red_row["Color_Red"] == 1
    assert red_row["Color_Blue"] == 0


def test_onehot_encoder_multiple_columns():

    X = pd.DataFrame({
        "Color": ["Red", "Blue"],
        "Size": ["S", "M"]
    })

    enc = OneHotEncoder()
    enc.fit(X)

    Xt = enc.transform(X)

    assert "Color_Red" in Xt.columns
    assert "Size_S" in Xt.columns


# =====================================================
# SimpleImputer
# =====================================================

def test_imputer_mean():

    X = pd.DataFrame({
        "A": [1, np.nan, 3]
    })

    imp = SimpleImputer(strat="mean")
    imp.fit(X)

    Xt = imp.transform(X)

    assert not Xt.isna().any().any()
    assert np.isclose(Xt.loc[1, "A"], 2.0)


def test_imputer_median():

    X = pd.DataFrame({
        "A": [1, np.nan, 100]
    })

    imp = SimpleImputer(strat="median")
    imp.fit(X)

    Xt = imp.transform(X)

    assert Xt.loc[1, "A"] == 50.5


def test_imputer_mode():

    X = pd.DataFrame({
        "A": [1, 1, np.nan, 2]
    })

    imp = SimpleImputer(strat="mode")
    imp.fit(X)

    Xt = imp.transform(X)

    assert Xt.loc[2, "A"] == 1


def test_imputer_removes_nans():

    X = pd.DataFrame({
        "A": [1, np.nan],
        "B": [np.nan, 5]
    })

    imp = SimpleImputer()
    imp.fit(X)

    Xt = imp.transform(X)

    assert not Xt.isna().any().any()


# =====================================================
# Error handling
# =====================================================

def test_transform_before_fit_scaler():

    scaler = StandardScaler()

    with pytest.raises(Exception):
        scaler.transform(pd.DataFrame({"A": [1]}))


def test_transform_before_fit_encoder():

    enc = OneHotEncoder()

    with pytest.raises(Exception):
        enc.transform(pd.DataFrame({"A": ["x"]}))


def test_transform_before_fit_imputer():

    imp = SimpleImputer()

    with pytest.raises(Exception):
        imp.transform(pd.DataFrame({"A": [1]}))