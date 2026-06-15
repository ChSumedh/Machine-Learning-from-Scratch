import numpy as np
import pytest
from mls.Ensemble import RandomForestClassifier,RandomForestRegressor

def test_classifier_fit_predict():

    X = np.array([
        [0],
        [0],
        [1],
        [1]
    ])

    y = np.array([0,0,1,1])

    rf = RandomForestClassifier(
        n_estimators=10
    )

    rf.fit(X,y)

    preds = rf.predict(X)

    assert preds.shape == y.shape
    assert np.mean(preds == y) >= 0.75

def test_classifier_predict_proba_shape():

    X = np.array([
        [0],
        [1]
    ])

    y = np.array([0,1])

    rf = RandomForestClassifier(
        n_estimators=5
    )

    rf.fit(X,y)

    probs = rf.predict_proba(X)

    assert probs.shape == (2,2)

def test_classifier_probability_sum_to_one():

    X = np.random.rand(50,3)

    y = (X[:,0] > 0.5).astype(int)

    rf = RandomForestClassifier(
        n_estimators=5
    )

    rf.fit(X,y)

    probs = rf.predict_proba(X)

    assert np.allclose(
        probs.sum(axis=1),
        1.0
    )

def test_classifier_number_of_trees():

    X = np.random.rand(100,4)

    y = (X[:,0] > 0.5).astype(int)

    rf = RandomForestClassifier(
        n_estimators=17
    )

    rf.fit(X,y)

    assert len(rf.trees) == 17

def test_classifier_learns_signal():

    np.random.seed(42)

    X = np.random.rand(300,5)

    y = (
        X[:,0] + X[:,1] > 1
    ).astype(int)

    rf = RandomForestClassifier(
        n_estimators=25
    )

    rf.fit(X,y)

    preds = rf.predict(X)

    acc = np.mean(preds == y)

    assert acc > 0.9

def test_regressor_fit_predict_shape():

    X = np.random.rand(100,3)

    y = np.random.rand(100)

    rf = RandomForestRegressor(
        n_estimators=10
    )

    rf.fit(X,y)

    preds = rf.predict(X)

    assert preds.shape == y.shape

def test_regressor_number_of_trees():

    X = np.random.rand(50,2)

    y = np.random.rand(50)

    rf = RandomForestRegressor(
        n_estimators=23
    )

    rf.fit(X,y)

    assert len(rf.trees) == 23

def test_regressor_learns_signal():

    np.random.seed(42)

    X = np.random.rand(500,3)

    y = (
        2*X[:,0]
        + 3*X[:,1]
        - X[:,2]
    )

    rf = RandomForestRegressor(
        n_estimators=30
    )

    rf.fit(X,y)

    preds = rf.predict(X)

    mse = np.mean(
        (preds - y) ** 2
    )

    assert mse < 0.2

def test_regressor_constant_target():

    X = np.random.rand(100,4)

    y = np.ones(100) * 5

    rf = RandomForestRegressor(
        n_estimators=10
    )

    rf.fit(X,y)

    preds = rf.predict(X)

    assert np.allclose(
        preds,
        5,
        atol=1e-6
    )