import numpy as np
import math
import utils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt

def main():

    # training
    X_train, X_val, X_test, y_train, y_val, y_test = utils.getData("white")
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    pred = LR.predict(X_train)
    pred = np.rint(pred)
    y_train = np.rint(np.array(y_train))
    weights = LR.coef_
    print("Training:")
    print("Mean squared error: " + str(mean_squared_error(y_train, pred)))
    print("Mean absolute error: " + str(mean_absolute_error(y_train, pred)))
    print("Accuracy: " + str(utils.accuracy(pred, y_train)))

    # validation
    LR = LinearRegression()
    best_err = 10000000
    num_batches = 10
    batch_size = math.floor((len(y_val) / num_batches))
    for i in range(num_batches):

        X_batch = X_val[i*batch_size:(i+1)*batch_size]
        y_batch = y_val[i*batch_size:(i+1)*batch_size]

        model = LR.fit(X_batch, y_batch)
        pred = LR.predict(X_batch)
        pred = np.rint(pred)
        y_batch = np.rint(np.array(y_batch))

        err = mean_squared_error(y_batch, pred)
        if err < best_err:
            best_err = err
            best_model = model


    # testing
    pred = best_model.predict(X_test)
    pred = np.rint(pred)
    y_test = np.rint(np.array(y_test))
    print("Testing:")
    print("Mean squared error: " + str(mean_squared_error(y_test, pred)))
    print("Mean absolute error: " + str(mean_absolute_error(y_test, pred)))
    print("Accuracy: " + str(utils.accuracy(pred, y_test)))

main()

