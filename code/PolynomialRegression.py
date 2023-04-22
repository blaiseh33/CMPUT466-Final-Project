from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import utils
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt

poly_degree = 2

def main():

    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)

    # training
    X_train, X_val, X_test, y_train, y_val, y_test = utils.getData("red")
    poly_features = poly.fit_transform(X_train)
    LR = LinearRegression()
    LR.fit(poly_features, y_train)
    pred = LR.predict(poly_features)
    pred = np.rint(pred)
    y_train = np.rint(np.array(y_train))
    print("Training:")
    print("Mean squared error: " + str(mean_squared_error(y_train, pred)))
    print("Mean absolute error: " + str(mean_absolute_error(y_train, pred)))
    print("Accuracy: " + str(utils.accuracy(pred, y_train)))

    # validation
    best_err = 10000000
    num_batches = 1
    batch_size = math.floor((len(y_val) / num_batches))
    for i in range(num_batches):

        X_batch = X_val[i*batch_size:(i+1)*batch_size]
        y_batch = y_val[i*batch_size:(i+1)*batch_size]
        
        poly_features = poly.fit_transform(X_batch)
        model = LR.fit(poly_features, y_batch)
        pred = LR.predict(poly_features)
        pred = np.rint(pred)
        y_batch = np.rint(np.array(y_batch))

        err = mean_squared_error(y_batch, pred)
        if err < best_err:
            best_err = err
            best_model = model

    # testing
    poly_features = poly.fit_transform(X_test)
    pred = best_model.predict(poly_features)
    pred = np.rint(pred)
    y_test = np.rint(np.array(y_test))
    print("Testing")
    print("Mean squared error: " + str(mean_squared_error(y_test, pred)))
    print("Mean absolute error: " + str(mean_absolute_error(y_test, pred)))
    print("Accuracy: " + str(utils.accuracy(pred, y_test)))
    print("Close Accuracy (+-1 score): " + str(utils.close_accuracy(pred, y_test)))

main()


