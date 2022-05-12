from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
from Knotilus.Knotilus import Knotilus
import pandas as pd
import numpy as np

# Load and transform data
df = pd.read_csv('./data/pw_data3_5000.csv')
ss  = MinMaxScaler()
foo = ss.fit_transform(df)
synthetic_500 = pd.DataFrame(foo)

# Scale the data from 0 to 1
ss  = MinMaxScaler()
covid = ss.fit_transform(df)
covid = pd.DataFrame(covid)

fullVariable = np.array(covid[0])
fullTarget   = np.array(covid[1])

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
X_train = fullVariable[trainIndices]
y_train = fullTarget[trainIndices]
X_test  = fullVariable[testIndices]
y_test  = fullTarget[testIndices]

print('Starting the model')
# Train model
model = Knotilus(X_train, y_train)
model = model.fit(numKnots=4)

# Plot the resulting model
plt.title('Auto Knot Selection Example')
plt.scatter(covid[0], covid[1])
plt.plot(X_test, model.predict(X_test), 'r')
plt.show()