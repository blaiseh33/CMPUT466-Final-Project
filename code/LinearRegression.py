import numpy as np
import utils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = utils.getData("white")
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    pred = LR.predict(X_test)
    pred = np.rint(pred)
    y_test = np.rint(np.array(y_test))
    print(LR.coef_)
    print("Mean squared error: " + str(mean_squared_error(y_test, pred)))
    print("Mean absolute error: " + str(mean_absolute_error(y_test, pred)))
    print("Accuracy: " + str(utils.accuracy(pred, y_test)))

main()

