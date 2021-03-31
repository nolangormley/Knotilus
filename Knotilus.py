from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numdifftools as nd
import pyswarms as ps
import pandas as pd
import numpy as np
import sys

class Knotilus:

    def __init__(self, X, y, useMax=False):
        self.variable  = np.array(X)
        self.target    = np.array(y)
        self.knotLoc   = []
        self.useMax    = useMax
        self.SofterMaxVec = np.vectorize(self.SofterMax, otypes=[np.float64])

    def fit(self, numKnots=2, optim='Nelder-Mead', error=None, alpha=1e-8, gamma=1e-3, alphaGamma=0, tolerance=None, alphaLoop=False, step=1e-2, minAlpha=1e-8, verbose=False, logging=False):
        self.optim      = optim
        self.alpha      = alpha
        self.gamma      = gamma
        self.alphaGamma = alphaGamma
        self.tolerance  = tolerance
        self.alphaLoop  = alphaLoop
        self.step       = step
        self.minAlpha   = minAlpha
        self.verbose    = verbose
        self.logging    = logging
        self.score      = None

        # Create log if logging is enabled
        if self.logging:
            self.log = []

        # Set correct error function
        if error is None:
            self.errFunc = self.SSE
        elif callable(error):
            self.errFunc = error
        else:
            print('Error function not supported')
            return

        if numKnots == 'auto':
            self.autoKnots = True
            self.numKnots  = 1
        else:
            # If not auto-knot number selection, fit single model and return result
            self.autoKnots = False
            self.numKnots  = numKnots
            if self.alphaLoop:
                self.fitAlphaLoop()
            else:
                self.fitSingle()
            return self

        while True:
            # Run model
            if self.alphaLoop:
                self.fitAlphaLoop()
            else:
                self.fitSingle()

            # If the model scores within threshold better than the last, keep going
            if self.score == None or self.errFunc(self.predict_(), self.target) / self.score <= 1 - self.gamma:
                # Store values of best model
                self.bestModel = (self.knotLoc, self.coef)
                self.score     = self.errFunc(self.predict_(), self.target)
                # Set new knot val and continue
                self.numKnots += 1
                self.knotLoc   = []
            else:
                # return the previous model as the current one did not meet criteria
                # return self.bestModel
                self.numKnots -= 1
                self.knotLoc   = self.bestModel[0]
                self.coef      = self.bestModel[1]
                defaultMax     = self.useMax
                self.useMax    = True
                self.knots     = self.CreateKnots(self.variable, self.knotLoc)
                self.useMax    = defaultMax

                return self

    def fitAlphaLoop(self):
        self.alphaScore = np.inf

        alphaVals = np.arange(1, self.minAlpha, -1 * self.step)
        alphaVals = np.append(alphaVals, self.minAlpha)

        # print('AlphaVals:', alphaVals)

        for alpha in alphaVals:
            self.alpha = alpha
            self.fitSingle()
            # print('Alpha:', self.alpha, 'Error:', self.errFunc(self.predict_(), self.target))

            if self.errFunc(self.predict_(), self.target) / self.alphaScore > 1 - self.alphaGamma:
                break

            self.alphaScore = self.errFunc(self.predict_(), self.target)

        self.alpha += self.step
        self.fitSingle()

    def fitSingle(self):
        # Create initial estimates of the knot locations
        # This evenly spaces the knots across the dataset
        self.knotLoc = []
        for i in range(self.numKnots):
            self.knotLoc.append(self.variable[int(i * (self.variable.shape[0] / self.numKnots))])

        # Create the representation of knots through the dataset
        self.knots = self.CreateKnots(self.variable, self.knotLoc)

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
        elif self.optim == 'pso':
            bounds = (np.zeros(self.numKnots), np.ones(self.numKnots))

            # Set-up hyperparameters
            options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
            # Call instance of PSO
            optimizer = ps.single.GlobalBestPSO(
                n_particles=5,
                dimensions=self.numKnots,
                options=options,
                bounds=bounds
            )
            # Perform optimization
            cost, pos = optimizer.optimize(self.error, iters=200, verbose=self.verbose)

            self.knotLoc = pos
        elif self.optim == 'constrained':
            self.results = minimize(
                self.error,
                self.knotLoc,
                method = 'SLSQP',
                tol = self.tolerance,
                bounds=bounds
            )
            self.knotLoc = self.results.x
        elif self.optim == 'gradient':
            self.results = minimize(
                self.error,
                self.knotLoc,
                method = 'trust-exact',
                tol = self.tolerance,
                hess=self.Hessian,
                jac=self.Jacobian
            )
            self.knotLoc = self.results.x
        elif self.optim == 'withCoef':
            self.results = minimize(
                self.errorCoef,
                np.append(self.knotLoc, self.coef),
                method = 'Nelder-Mead',
                tol = self.tolerance
            )
            self.knotLoc = self.results.x
        else:
            self.results = minimize(
                self.error,
                self.knotLoc,
                method = self.optim,
                tol = self.tolerance
            )
            self.knotLoc = self.results.x

        # Run final model
        defaultMax       = self.useMax
        self.useMax      = True
        self.knots       = self.CreateKnots(self.variable, self.knotLoc)
        self.useMax      = defaultMax
        self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.target)
        self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)

    # Error function for the piecewise model
    def error(self, X):
        # Get the error of this estimate of the knot placements
        if X.ndim > 1:
            error = []
            for x in X:
                self.iterations += 1
                # Run linear model based on the given knot placements to get the coefficients
                self.knots       = self.CreateKnots(self.variable, x)
                self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.target)
                self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)
                error.append(self.errFunc(self.predict_(), self.target))

                # Print out status of model
                if self.verbose:
                    print(f'\rKnot: {self.numKnots}\tIteration: {self.iterations}\nError: {sum(error)}\tAlpha: {self.alpha}', end = '\r')

                # Add to log
                if self.logging:
                    self.log.append({
                        'error' : error[-1],
                        'knotLoc' : x,
                        'coef' : self.coef
                    })
        else:
            self.iterations += 1
            # Run linear model based on the given knot placements to get the coefficients
            self.knots       = self.CreateKnots(self.variable, X)
            self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.target)
            self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)
            error = self.errFunc(self.predict_(), self.target)

            # Print out status of model
            if self.verbose:
                print(f'\rKnot: {self.numKnots}   Iteration: {self.iterations}   Error: {error}   Alpha: {self.alpha}', end = '\r')

            # Add to log
            if self.logging:
                self.log.append({
                    'error' : error,
                    'knotLoc' : X,
                    'coef' : self.coef
                })

        return error

    def errorCoef(self, X):
        self.iterations += 1
        # Run linear model based on the given knot placements to get the coefficients
        self.knots       = self.CreateKnots(self.variable, X[:self.numKnots])
        self.coef        = X[self.numKnots:]
        error = self.errFunc(self.predict_(), self.target)

        # Print out status of model
        if self.verbose:
            print(f'\rKnot: {self.numKnots}   Iteration: {self.iterations}   Error: {error}   Alpha: {self.alpha}', end = '\r')

        # Add to log
        if self.logging:
            self.log.append({
                'error' : error,
                'knotLoc' : X,
                'coef' : self.coef
            })

        return error

    # Creates the numerical representation of the knots for each observation
    def CreateKnots(self, X, knotLoc):
        if self.useMax:
            knots = np.repeat(X, len(knotLoc)).reshape(X.shape[0], len(knotLoc))
            return np.maximum(knots-knotLoc, np.zeros(knots.shape))
        else:
            knots = np.repeat(X, len(knotLoc)).reshape(X.shape[0], len(knotLoc))
            return self.SofterMaxVec(knots-knotLoc)

    # Polynomial approximation of a max function
    def SofterMax(self, x):
        if x > 0 and x <= self.alpha:
            return round(3 * self.alpha**(-4) * x**5 - 8 * self.alpha**(-3) * x**4 + 6 * 
                    self.alpha**(-2) * x**3, 8)
        elif x > self.alpha:
            return x
        else:
            return 0

    def MSE(self, yPred, yTrue):
        return np.sum((yTrue - yPred)**2)/self.variable.shape[0]

    def RMSE(self, yPred, yTrue):
        return np.sqrt(self.MSE(yPred, yTrue))

    def SSE(self, yPred, yTrue):
        return np.sum((yTrue - yPred)**2)

    def predict_(self, max=False):
        return np.dot(self.knots, self.coef[1:]) + self.coef[0]

    def predict(self, max=False):
        self.knots       = self.CreateKnots(self.variable, self.knotLoc)
        self.linearModel = LinearRegression(n_jobs=-1).fit(self.knots, self.target)
        self.coef        = np.append(self.linearModel.intercept_, self.linearModel.coef_)
        return np.dot(self.knots, self.coef[1:]) + self.coef[0]

    def Hessian(self, X):
        return nd.Hessian(lambda x: self.error(x))(X)

    def Jacobian(self, X):
        return nd.Jacobian(lambda x: self.error(x))(X).ravel()

if __name__ == "__main__":
    df = pd.read_csv('./src/data/us_covid19_daily.csv')
    df['deathIncrease'] = df['deathIncrease'].astype(int)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['unixTime'] = df['date'].astype(int) / 10**9
    df = df[['unixTime', 'deathIncrease']]

    ss  = MinMaxScaler()
    foo = ss.fit_transform(df)
    foo = pd.DataFrame(foo)

    start = timer()
    model = Knotilus(foo[0], foo[1])

    model = model.fit(numKnots=6, error=model.SSE, optim='withCoef', verbose=True)
    
    end = timer()


    print('\n\nTime:', end - start)
    print('Alpha:', model.alpha)

    knotVals = np.sort(model.knotLoc)
    for knot in knotVals:
        print('Knot:', knot)

    plt.title('Auto Knot Selection Example')
    plt.scatter(foo[0], foo[1])
    plt.plot(foo[0], model.predict(), 'r')

    plt.show()