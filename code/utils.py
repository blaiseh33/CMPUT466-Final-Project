import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# percentage of data to be split into val and test data - remainder is training
num_val_test = 0.3

# features and target
features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
target = 'quality'

def getData(wine_type):
    if wine_type == "red":
        dataset = pd.read_csv("winequality-red.csv")
        dataset = dataset.sample(frac=1)
        X_all = dataset.drop("quality", axis=1)
        y_all = dataset['quality']
        X_train = X_all[1:1000]
        X_val = X_all[1000:1300]
        X_test = X_all[1300:1599]
        y_train = y_all[1:1000]
        y_val = y_all[1000:1300]
        y_test = y_all[1300:1599]
        #X_train, X_test_val, y_train, y_test_val = train_test_split(X_all, y_all, test_size=num_val_test, random_state=33)
        #X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.3, random_state=33)
    elif wine_type == "white":
        dataset = pd.read_csv("winequality-white.csv")
        dataset = dataset.sample(frac=1)
        #X_all = dataset.drop("quality", axis=1)
        #y_all = dataset['quality']
        X_all = dataset[features].values.reshape(-1, len(features))
        y_all = dataset[target].values
        X_train = X_all[1:3000]
        X_val = X_all[3000:4000]
        X_test = X_all[4000:4898]
        y_train = y_all[1:3000]
        y_val = y_all[3000:4000]
        y_test = y_all[4000:4898]
        #X_train, X_test_val, y_train, y_test_val = train_test_split(X_all, y_all, test_size=num_val_test, random_state=33)
        #X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.3, random_state=33)

    return X_train, X_val, X_test, y_train, y_val, y_test

def accuracy(pred, y_test):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == y_test[i]:
            correct += 1
    acc = correct / len(pred)
    return acc