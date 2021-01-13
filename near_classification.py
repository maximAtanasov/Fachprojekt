# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:38:51 2020

"""

import os
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import neural_network, dummy
import joblib
from data_loader import load_train, load_test_all


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return TP, FP, TN, FN

def trainModels():
    models = []
    for i in range(1, 24):
        if os.path.exists(f"models/model_{i}.sav"):
            models.append(joblib.load(f"models/model_{i}.sav"))
        else:
            features, labels = load_train(i)

            if i in [1, 22, 23]: #strips 1, 22, 13 sind immer near=0
                model = dummy.DummyClassifier(strategy='constant', constant=0)
            else:
                model = neural_network.MLPClassifier(hidden_layer_sizes=(300,), activation='tanh', solver='adam', max_iter=500, random_state=1)
            
            # Create and Shuffle Index
            permutation = np.random.permutation(features.shape[0])
            train_ratio = 0.8

            # Create 10 Splits
            splits = np.array_split(permutation, [np.int64(train_ratio * permutation.shape[0]), ])
            X_train = features[splits[0]]
            X_test = features[splits[1]]

            y_train = labels[splits[0]]
            y_test = labels[splits[1]]

            #TODO:
            if np.max(y_train) == 0.0:
                y_train[0] = 1.0

            model.fit(X_train, y_train)

            joblib.dump(model, f"models/model_{i}.sav")
            models.append(model)
            predictions = model.predict(X_test)
            # Divide the number of correct predictions by the test data size
            accuracy = np.where(predictions == y_test)[0].shape[0] / y_test.shape[0]
            print("Accuracy: {:.4f} %".format(accuracy * 100))

            try:
                TP, FP, TN, FN = perf_measure(y_test, predictions)
                print(f"true positive {TP / (TP + FP)}")
            except Exception as e:
                print(f"Error: {str(e)}")
    
    return models

def computeStrips(models, frames):
    predictions = []
    assert len(frames) == (3412 * len(models))
    for i in range(0, 3412):
        confidence = []
        for j in range(0, len(models)):
            index = i + j * 3412
            conf = models[j].predict_proba([frames[index]])[0][1] #confidence value for this strip
            confidence.append(conf)

        predictions.append(confidence)

    for i in range(len(predictions)):
        sortedPred = sorted(predictions[i], reverse=True)
        strips = []
        if sortedPred[0] > 0.85:
            strips.append(np.argmax(predictions[i])+1)
            predictions[i][strips[0]-1] = 0
            if abs(sortedPred[0] - sortedPred[1]) < 0.20:
                strips.append(np.argmax(predictions[i])+1)
                predictions[i][strips[0]-1] = 0

        predictions[i] = sorted(strips)

    avg = lambda l: sum(l) / float(len(l))
    for i in range(len(predictions)):
        if len(predictions[i]) != 0:
            continue
        assert i > 0 and i < len(predictions)-1
        before = predictions[i-1]
        afterIdx = i+1
        while len(predictions[afterIdx]) == 0:
            afterIdx += 1
        after = predictions[afterIdx]

        assert len(before) == 1 or abs(before[0] - before[1]) == 1
        assert len(after) == 1 or abs(after[0] - after[1]) == 1

        m = (avg(after) - avg(before)) / (afterIdx - (i-1))
        b = avg(after) - m * afterIdx
        for j in range(i, afterIdx):
            predStrip = m*j+b
            middle = predStrip%1
            if middle < 0.4:
                predStrip = [math.floor(predStrip)]
            elif middle > 0.6:
                predStrip = [math.ceil(predStrip)]
            else:
                predStrip = [math.floor(predStrip), math.ceil(predStrip)]
            predictions[j] = predStrip
    
    for i in range(len(predictions)):
        pred = predictions[i]
        origPred = list(pred)
        assert len(pred) == 1 or len(pred) == 2
        if len(pred) == 2 and abs(pred[0] - pred[1]) > 1:
            before = predictions[i-1]
            after = predictions[i+1]
            assert len(after) == 1 or abs(after[0] - after[1]) == 1

            avgBefore = avg(before)
            avgAfter = avg(after)
            for p in pred:
                if abs(avgBefore - p) >= 1.5 and abs(avgAfter - p) >= 1.5:
                    pred.remove(p)

    return predictions

if __name__ == "__main__":
    models = trainModels()
    frames = load_test_all()
    predictions = computeStrips(models, frames)

    stripsDict = defaultdict(list)
    for pred in predictions:
        for strip in range(1, len(models)+1):
            if strip in pred:    
                stripsDict[strip].append(1)
            else:
                stripsDict[strip].append(0)

    predictions = []
    for strip in range(1, len(models)+1):
        assert len(stripsDict[strip]) == 3412
        predictions.extend(stripsDict[strip])

    csv = {
        'Id': list(range(len(predictions))),
        'Predicted': predictions
    }
    df = pd.DataFrame(csv, columns=['Id', 'Predicted'])
    df.to_csv("predictions.csv", index = False)
