import numpy as np

 # Error Functions
def MSE(yPred, yTrue):
    return np.sum((yTrue - yPred)**2)/yTrue.shape[0]

def RMSE(yPred, yTrue):
    return np.sqrt(MSE(yPred, yTrue))

def SSE(yPred, yTrue):
    return np.sum((yTrue - yPred)**2)