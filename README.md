# Knotilus: A Differentiable Piecewise Linear Regression Framework

This is the main repository for the Knotilus piecewise linear regression package. The past works on this method including many tests used in the creation of the thesis surrounding this package are located on the [Gitlab repository](https://gitlab.com/nolangormley/nolan-gormley-bgsu-thesis).

The whole package of Knotilus is held within the file Knotilus.py. An example use of this package is shown in example.py. 

Pull requests are always welcomed. Feel free to contact me at nolangormley@gmail.com with any questions.

## Basic Usage

### Data Preparation
This method works on data that is standardized with between 0 and 1, we recommend the Sci-Kit Learn MinMaxScaler.

```
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('./data/pw_data3_5000.csv')

# Scale the data from 0 to 1
ss  = MinMaxScaler()
df = ss.fit_transform(df)
df = pd.DataFrame(df)
```

### Creating and training the model
Split the data in a way that best suits your case. The Knotilus.fit function has many parameters as shown in the Knotilus.py file.
```
# Train model
model = Knotilus(X_train, y_train)
model = model.fit(numKnots='auto')
```

### Predicting with a trained model
Set the model's variables to those that you want to predict, then use the model.predict function to predict the values. This will be changed in the future to act more like an Sci-Kit Learn regression package where the values are passed into a predict function.
```
# Set data to testing data
model.variable = X_test
model.target   = y_test

# Plot the resulting model
plt.title('Auto Knot Selection Example')
plt.scatter(covid[0], covid[1])
plt.plot(X_test, model.predict(), 'r')
plt.show()
```
