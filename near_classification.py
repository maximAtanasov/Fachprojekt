# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:38:51 2020

"""

import os
import pandas as pd
import numpy as np
from sklearn import svm
import joblib
from data_loader import load_train, load_test_all

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

if os.path.exists("model.sav"):
    p = joblib.load("model.sav")
else:
    features, labels = load_train(2)

    p = svm.LinearSVC(max_iter=3000)
    # Create and Shuffle Index
    permutation = np.random.permutation(features.shape[0])
    train_ratio = 0.8
    # Create 10 Splits
    splits = np.array_split(permutation, [np.int64(train_ratio*permutation.shape[0]),])
    X_train = features[splits[0]]
    X_test = features[splits[1]]

    y_train = labels[splits[0]]
    y_test = labels[splits[1]]

    p.fit(X_train, y_train)

    joblib.dump(p, "model.sav")

    predictions = p.predict(X_test)
    # Divide the number of correct predictions by the test data size
    accuracy = np.where(predictions==y_test)[0].shape[0] / y_test.shape[0]
    print("Accuracy: {:.4f} %".format(accuracy*100))

    TP, FP, TN, FN = perf_measure(y_test, predictions)
    print(f"true positive {TP / (TP + FP)}")
    print(f"false positive {FP / (TP + FP)}")

frames = load_test_all()

predictions = []
for i in range(len(frames)):
    strip = int(i / 3412)
    model = p
    predictions.append(model.predict([frames[i]])[0])

csv = {
    'Id': [i for i in range(len(predictions))],
    'Predicted': predictions
}

df = pd.DataFrame (csv, columns = ['Id','Predicted'])
df.to_csv("predictions.csv")