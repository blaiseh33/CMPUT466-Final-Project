import numpy as np
import utils
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt

def main():

    # Hyperparameters
    NN = MLPRegressor(
        activation = 'relu',
        hidden_layer_sizes = (100,),
        alpha = 0.001,
        random_state = 20,
        early_stopping = False
    )

    # training
    X_train, X_val, X_test, y_train, y_val, y_test = utils.getData("red")
    NN.fit(X_train, y_train)
    pred = NN.predict(X_train)
    pred = np.rint(pred)
    y_train = np.rint(np.array(y_train))
    print("Training:")
    print("Mean squared error: " + str(mean_squared_error(y_train, pred)))
    print("Mean absolute error: " + str(mean_absolute_error(y_train, pred)))
    print("Accuracy: " + str(utils.accuracy(pred, y_train)))

    # validation
    best_model = None
    epochs = 50
    best_err = 10000000
    for i in range(epochs):
        model = NN.fit(X_val, y_val)
        pred = model.predict(X_val)
        pred = np.rint(pred)
        y_val = np.rint(np.array(y_val))
        err = mean_squared_error(y_val, pred)
        if err < best_err:
            best_model = model
            best_err = err

    # testing
    pred = best_model.predict(X_test)
    pred = np.rint(pred)
    y_test = np.rint(np.array(y_test))
    print("Testing:")
    print("Mean squared error: " + str(mean_squared_error(y_test, pred)))
    print("Mean absolute error: " + str(mean_absolute_error(y_test, pred)))
    print("Accuracy: " + str(utils.accuracy(pred, y_test)))
    print("Close Accuracy (+-1 score): " + str(utils.close_accuracy(pred, y_test)))

main()