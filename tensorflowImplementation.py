from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from scipy.optimize import minimize, differential_evolution
from sklearn.preprocessing import MinMaxScaler
from numdifftools import Hessian, Jacobian
from sklearn.model_selection import KFold
from timeit import default_timer as timer
import matplotlib.pyplot as plt
# from PSO import optimize
import pandas as pd
import numpy as np
import copy

# import pyswarms as ps
# from pyswarms.utils.functions import single_obj as fx

import tensorflow as tf
import tensorflow_probability as tfp
# import tf.experimental.numpy as tnp


class Knotilus:

    def __init__(self, X, y):
        self.variable = tf.convert_to_tensor(X)
        self.target   = tf.convert_to_tensor(y)
        self.knotLoc  = tf.convert_to_tensor([])

        # self.SofterMaxVec = tf.map_fn(self.SofterMax, otypes=[np.float64])

    def predict(self):
        return np.dot(self.knots, self.coef[1:]) + self.coef[0]

    def SofterMax(self, x):
        if x > 0 and x <= self.alpha:
            return 3 * self.alpha**(-4) * x**5 - 8 * self.alpha**(-3) * x**4 + 6 * self.alpha**(-2) * x**3
        elif x > self.alpha:
            return x
        else:
            return 0.

    @tf.function
    def softVec(self, x):
        X = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in range(len(x)):
            X = X.write(i, self.SofterMax(x[i]))
        return X.stack()


    # Error function for the piecewise model
    def SquaredError(self, X):
        # Get the error of this estimate of the knot placements
        if X.ndim > 1:
            sse = []
            for x in X:
                self.iterations += 1
                # Run linear model based on the given knot placements to get the coefficients
                self.knots       = self.CreateKnots(self.variable, x)
                self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.target)
                self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)
                sse.append(tf.reduce_sum((self.target - self.predict())**2))

                # Print out status of model
                if self.verbose:
                    print(f'\rKnot: {self.numKnots}   Iteration: {self.iterations}   SSE: {sum(sse)}', end = '\r')
            sse = tf.convert_to_tensor(sse)
        else:
            self.iterations += 1
            # Run linear model based on the given knot placements to get the coefficients
            self.knots       = self.CreateKnots(self.variable, X)
            self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.target)
            self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)
            sse = tf.reduce_sum((self.target - self.predict())**2)

            # Print out status of model
            if self.verbose:
                print(f'\rKnot: {self.numKnots}   Iteration: {self.iterations}   SSE: {sse}', end = '\r')

        return sse

    # # Error function for the piecewise model
    # def SquaredError(self, x):
    #     # Run linear model based on the given knot placements to get the coefficients
    #     self.knots       = self.CreateKnots(self.variable, x)
    #     self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.target)
    #     self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)

    #     self.iterations += 1

    #     # Get the error of this estimate of the knot placements
    #     sse = tf.reduce_sum((self.target - self.predict())**2)

    #     # Print out status of model
    #     if self.verbose:
    #         print(f'\rKnot: {self.numKnots}   Iteration: {self.iterations}   SSE: {sse}', end = '\r')

    #     return sse


    # Created the numerical representation of the knots for each observation
    def CreateKnots(self, X, knotLoc):
        knots = tf.repeat(X, knotLoc.shape[0])
        knots = tf.reshape(knots, (X.shape[0], knotLoc.shape[0]))
        knots = tf.reshape(knots - knotLoc, (X.shape[0] * knotLoc.shape[0]))
        knots = tf.map_fn(self.SofterMax, knots, parallel_iterations=1000)
        knots = tf.reshape(knots, (X.shape[0], knotLoc.shape[0]))
        return knots

    def fit_cv(self, numKnots=2, folds=6, optim='Nelder-Mead', alpha=1e-8, gamma=1e-3, tolerance=None, verbose=False):
        variable = self.variable
        target   = self.target

        results  = []
        bestResult = None
        kf = KFold(n_splits=folds)
        kf.get_n_splits(self.variable)

        for train_index, test_index in kf.split(self.variable):
            self.variable = variable[train_index]
            self.target   = target[train_index]

            model = self.fit(numKnots=numKnots, optim=optim, alpha=alpha, gamma=gamma, tolerance=tolerance, verbose=verbose)
            model.variable = variable[test_index]
            model.target   = target[test_index]
            model.knots    = model.CreateKnots(model.variable, model.knotLoc)

            modelResult = {
                'sse'      : model.SquaredError(model.knotLoc),
                'numKnots' : model.numKnots,
                'knotLoc'  : model.knotLoc,
            }

            print(model.coef)

            if bestResult is None or modelResult['sse'] < bestResult['sse']:
                bestResult = modelResult

            results.append(modelResult)
        
        self.variable    = variable
        self.target      = target
        self.numKnots    = bestResult['numKnots']
        self.knotLoc     = bestResult['knotLoc']
        self.knots       = self.CreateKnots(self.variable, self.knotLoc)
        self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.target)
        self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)
        self.cvresults   = results

        return self

    def fit(self, numKnots=2, optim='Nelder-Mead', alpha=1e-8, gamma=1e-3, tolerance=None, verbose=False):
        self.tolerance = tolerance
        self.optim     = optim
        self.alpha     = alpha
        self.gamma     = gamma
        self.score     = None
        self.verbose   = verbose

        if numKnots == 'auto':
            self.autoKnots = True
            self.numKnots  = 1
        else:
            self.autoKnots = False
            self.numKnots  = numKnots
            self.fit_single()
            return self

        while True:
            # Run model
            self.fit_single()

            # If the model scores within threshold better than the last, keep going
            if self.score == None or self.score / self.linearModel.score(self.knots, self.target) <= 1 - self.gamma:
                # Deep copy the current model so you don't have to recalulate when you go too far
                # self.bestModel = copy.deepcopy(self)
                self.bestModel = (self.knotLoc, self.coef)
                self.score     = self.linearModel.score(self.knots, self.target)
                # Set new knot val and continue
                self.numKnots += 1
                self.knotLoc  = []
            else:
                # return the previous model as the current one did TypeError: 'retval_' has dtype float64 in the main branch, but dtype float32 in the else branchot meet criteria
                # return self.bestModel
                self.numKnots -= 1
                self.knotLoc   = self.bestModel[0]
                self.coef      = self.bestModel[1]
                self.knots     = self.CreateKnots(self.variable, self.knotLoc)
                return self

    def fit_single(self):
        # Create initial estimates of the knot locations
        # This evenly spaces the knots across the dataset
        knotInd = []
        for i in range(self.numKnots):
            knotInd.append(int(i * (self.variable.shape[0] / self.numKnots)))
            # self.knotLoc.concat(self.variable[int(i * (self.variable.shape[0] / self.numKnots))])
        self.knotLoc = tf.gather(self.variable, knotInd)

        # Create the representation of knots through the dataset
        self.knots       = self.CreateKnots(self.variable, self.knotLoc)

        # Run an linear model to get the initial coefficients
        self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.target)
        self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)
        self.iterations  = 0

        # Create list of bounds to constrain the optimization of the knots from 0 to 1
        bounds = []
        for i in range(self.numKnots):
            bounds.append((0,1))

        # Run optimization routine on only the knot placement
        if self.optim == 'diffev':
            self.results = tfp.optimizer.differential_evolution_minimize(self.SquaredError, initial_position=self.knotLoc, population_size=40, population_stddev=2.0)
            self.knotLoc = self.results.position
        else:
            self.results = tfp.optimizer.nelder_mead_minimize(self.SquaredError, initial_vertex=self.knotLoc, parallel_iterations=1000)
            self.knotLoc = self.results.position

        # Run final model
        # self.knotLoc     = self.results.x
        self.knots       = self.CreateKnots(self.variable, self.knotLoc)
        self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.target)
        self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)

if __name__ == "__main__":
    df = pd.read_csv('./src/data/us_covid19_daily.csv')
    df['deathIncrease'] = df['deathIncrease'].astype(int)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['unixTime'] = df['date'].astype(int) / 10**9
    df = df[['unixTime', 'deathIncrease']]

    ss  = MinMaxScaler()
    foo = ss.fit_transform(df)
    foo = pd.DataFrame(foo)

    model = Knotilus(foo[0], foo[1])

    model.alpha = 1e-8
    model.SofterMax(model.variable)

    model.fit(numKnots=6, optim='diffev', verbose=True)

    print(model.knotLoc)

    plt.title('Auto Knot Selection Example')
    plt.scatter(foo[0], foo[1])
    plt.plot(foo[0], model.predict(), 'r')

    plt.show()