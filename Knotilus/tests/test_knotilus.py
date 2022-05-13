from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ..Knotilus import Knotilus
from sklearn import datasets
import pandas as pd
import numpy as np
import pytest

def initialize_model(logging = False):
    model = Knotilus(
        numKnots=2,
        optim='Nelder-Mead',
        error=None,
        alpha=1e-8,
        gamma=1e-3,
        tolerance=None,
        step=1e-2,
        verbose=False,
        logging=logging
    )

    return model

def get_dataset():
    diabetes = datasets.load_diabetes()
    ss  = MinMaxScaler()
    data = ss.fit_transform(np.column_stack((diabetes['data'][:,6], diabetes['target'])))

    X = data[:,0]
    y = data[:,1]

    return train_test_split(X, y, test_size=0.2)

def test_model_parameters():
    model = initialize_model()

    assert model.numKnots == 2
    assert model.optim == 'Nelder-Mead'
    assert model.alpha == 1e-8
    assert model.gamma == 1e-3
    assert model.tolerance is None
    assert model.step == 1e-2
    assert model.verbose == False
    assert model.logging == False

def test_model_fit():
    model = initialize_model()
    X_train, X_test, y_train, y_test = get_dataset()

    model.fit(X_train, y_train)
    assert model.linearModel is not None
    assert len(model.coef) - 1 == model.numKnots
    assert model.iterations > 0

def test_createknots():
    model = initialize_model()
    X_train, X_test, y_train, y_test = get_dataset()

    model.fit(X_train, y_train)

    assert model.CreateKnots(X_train, model.knotLoc).shape == (X_train.shape[0], model.numKnots)

def test_model_predict():
    model = initialize_model()
    X_train, X_test, y_train, y_test = get_dataset()

    model.fit(X_train, y_train)

    # Ensure the predicted vector is the same shape as the true outcome
    assert model.predict(X_test).shape == y_test.shape

def test_softermax():
    model = initialize_model()

    assert model.SofterMax(-1) == 0
    assert model.SofterMax(model.alpha) == model.alpha