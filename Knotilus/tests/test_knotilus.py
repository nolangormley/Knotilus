from sklearn.preprocessing import MinMaxScaler
from Knotilus import Knotilus
from sklearn import datasets
import pandas as pd
import numpy as np

class diabetes_test:
    diabetes = datasets.load_diabetes()
    ss  = MinMaxScaler()
    data = ss.fit_transform(np.column_stack((diabetes['data'][:,6], diabetes['target'])))

    X = data[:,0]
    y = data[:,1]

    model = Knotilus(
        numKnots=2,
        optim='Nelder-Mead',
        error=None,
        alpha=1e-8,
        gamma=1e-3,
        tolerance=None,
        step=1e-2,
        minAlpha=1e-8,
        verbose=False,
        logging=True
    )

    # Set up single bootstrap split
    np.random.seed(1)
    trainIndices = resample(
        np.arange(fullVariable.shape[0]),
        replace=True,
        n_samples=int(fullVariable.shape[0] * .8)
    )

    testIndices = np.array(
        [ind for ind in np.arange(fullVariable.shape[0]) if ind not in trainIndices]
    )

    X_train = X[trainIndices]
    y_train = y[trainIndices]
    X_test  = X[testIndices]
    y_test  = y[testIndices]

    def test_model_parameters(self):
        assert self.model.numKnots == 2
        assert self.model.optim == 'Nelder-Mead'
        assert self.model.error is None
        assert self.model.alpha == 1e-8
        assert self.model.gamma == 1e-3
        assert self.model.tolerance is None
        assert self.model.step == 1e-2
        assert self.model.minAlpha == 1e-8
        assert self.model.verbose == False
        assert self.model.logging == True
