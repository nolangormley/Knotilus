from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import LinearRegression
from .ModelLog import ModelLog
from .Metrics import *
import pandas as pd
import numpy as np

class Knotilus:

    def __str__(self):
        return f"\n\tError: {str(self.errFunc(self.predict(self.X), self.y))}\n" + \
               f"\tNum Knots: {str(self.numKnots)}\n" + \
               f"\tKnot Locations: {str(self.knotLoc)}\n" + \
               f"\tCoefs: {str(self.coef)}\n"

    def __init__(
        self,
        numKnots=2,
        optim='Nelder-Mead',
        error=None,
        alpha=1e-8,
        gamma=1e-3,
        tolerance=None,
        step=1e-2,
        verbose=False,
        logging=False
    ):
        self.numKnots     = numKnots
        self.optim        = optim
        self.alpha        = alpha
        self.gamma        = gamma
        self.tolerance    = tolerance
        self.step         = step
        self.verbose      = verbose
        self.logging      = logging
        self.score        = None
        self.knotLoc      = []
        self.SofterMaxVec = np.vectorize(self.SofterMax, otypes=[np.float64])

        # Set correct error function
        if error is None:
            self.errFunc = SSE
        elif callable(error):
            self.errFunc = error
        else:
            raise ValueError('Error function not supported')

    # User interface for fit method
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = np.array(X)
        self.y = np.array(y)

        # Create log if logging is enabled
        if self.logging:
            self.log = ModelLog(name_header = 'Knotilus')

        if self.numKnots != 'auto':
            # If not auto-knot number selection, fit single model and return result
            self.autoKnots = False
            self.__fit__()
            return self
        else:
            self.autoKnots = True
            self.numKnots  = 1

            # Add another knot the the model until the improvements are not within a given threshold better
            while self.score is None or self.errFunc(self.predict(self.X), self.y) / self.score <= 1 - self.gamma:
                
                # If the model scores within threshold better than the last, keep going
                if self.score is not None:
                    # Store values of best model
                    self.bestModel = (self.knotLoc, self.coef)
                    self.score     = self.errFunc(self.predict(self.X), self.y)
                    # Set new knot val and continue
                    self.numKnots += 1
                    self.knotLoc   = []

                # Run model
                self.__fit__()

            # return the previous model as the current one did not meet criteria
            self.numKnots -= 1
            self.knotLoc   = self.bestModel[0]
            self.coef      = self.bestModel[1]
            self.knots     = self.CreateKnots(self.X, self.knotLoc, useMax = True)

            return self

    # Backend fit method used to fit a model with N knots
    def __fit__(self):
        # Create initial estimates of the knot locations
        # This evenly spaces the knots across the dataset
        self.knotLoc = []
        for i in range(self.numKnots):
            self.knotLoc.append(int(i * (self.X.shape[0] / self.numKnots)))

        # Create the representation of knots through the dataset
        self.knots = self.CreateKnots(self.X, self.knotLoc)

        # Run an linear model to get the initial coefficients
        self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.y)
        self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)
        self.iterations  = 0

        # Create list of bounds to constrain the optimization of the knots from 0 to 1
        bounds = []
        for i in range(self.numKnots):
            bounds.append((0,1))

        # Run optimization routine on only the knot placement
        if self.optim == 'diffev':
            bounds = []

            for i in range(self.numKnots):
                bounds.append((0,1))

            self.results = differential_evolution(
                self.error,
                bounds,
                popsize=5,
                maxiter=40
            )
            self.knotLoc = self.results.x
        else:
            self.results = minimize(
                self.error,
                self.knotLoc,
                method = self.optim,
                tol = self.tolerance
            )
            self.knotLoc = []
            for res in self.results.x:
                self.knotLoc.append(self.X[int(res)])

        # Run final model
        self.knots       = self.CreateKnots(self.X, self.knotLoc, useMax = True)
        self.useMax      = False
        self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.y)
        self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)

    # Error function for the piecewise model
    def error(self, X: np.ndarray):
        # Get the error of this estimate of the knot placements
        if X.ndim > 1:
            error = []
            for x in X:
                self.iterations += 1
                # Run linear model based on the given knot placements to get the coefficients
                error.append(self.errFunc(self.predict(self.X, recalculate=True), self.y))

                # Print out status of model
                if self.verbose:
                    print(f'\rKnot: {self.numKnots}\tIteration: {self.iterations}\nError: {sum(error)}\tAlpha: {self.alpha}', end = '\r')

                # Add to log
                if self.logging:
                    self.log.logMessage(str(self))
        else:
            self.iterations += 1
            # Run linear model based on the given knot placements to get the coefficients
            self.knotLoc     = [self.X[int(knot)] for knot in X]
            error            = self.errFunc(self.predict(self.X, recalculate=True), self.y)

            # Print out status of model
            if self.verbose:
                print(f'\rKnot: {self.numKnots}   Iteration: {self.iterations}   Error: {error}   Alpha: {self.alpha}', end = '\r')

            # Add to log
            if self.logging:
                self.log.logMessage(str(self))

        return error
        
    # Creates the numerical representation of the knots for each observation
    def CreateKnots(self, X: np.ndarray, knotLoc: list, useMax = False) -> np.ndarray:
        # split single dimension dataset into a multi dimensional matrix to apply the piecewise function
        knots = np.repeat(X, len(knotLoc)).reshape(X.shape[0], len(knotLoc))

        # The max function is still used to produce the final model
        if useMax:
            return np.maximum(knots-knotLoc, np.zeros(knots.shape))
        else:
            return self.SofterMaxVec(knots-knotLoc)

    def predict(self, X: np.ndarray, recalculate = False, useMax = False) -> np.ndarray:
        self.knots = self.CreateKnots(X, self.knotLoc, useMax = useMax)
        if recalculate:
            self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.y)
            self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)
        return np.dot(self.knots, self.coef[1:]) + self.coef[0]

    # Polynomial approximation of a max function
    def SofterMax(self, x):
        if x > 0 and x <= self.alpha:
            return round(3 * self.alpha**(-4) * x**5 - 8 * self.alpha**(-3) * x**4 + 6 * 
                    self.alpha**(-2) * x**3, 8)
        elif x > self.alpha:
            return x
        else:
            return 0
            