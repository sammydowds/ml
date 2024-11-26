from ucimlrepo import fetch_ucirepo 
from logistic_regression import Model, gradient_descent

# Example usage of the logistic regression model class

# use adult income data
adult = fetch_ucirepo(id=2)

# pre process
adult.data.targets = adult.data.targets.replace({ "<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1 })
X = adult.data.features[['age', 'education-num', 'hours-per-week']][0:20]
y = adult.data.targets[0:11]
X_train = X[:10].to_numpy()
y_train = y[:10].to_numpy()
X_test = X[10:].to_numpy()
y_test = y[10:].to_numpy()

# learning rate and iterations 
alph = 0.1
iters = 10000

# train 
model = Model(alph, iters)
w, b = model.train(X_train, y_train) 
print(f"model trained: w:{w}, b:{b}")

# test prediction for single set of features 
print(f"testing: {X_test[0]}")
print(f"result should be: {y_test[0]}")
print("predicting...")
classification = model.predict(X_test[0])
print(classification)