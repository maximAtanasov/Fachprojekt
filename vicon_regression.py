import joblib
import numpy as np
from sklearn import svm, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

from data_loader import load_train, load_vicon_train, load_test_all


class DummyModel:
    def fit(self, X_train, X_test):
        pass
    def predict(self, X):
        return np.zeros(1)


models = [DummyModel()]
average_accuracy = 0
for i in range(2, 22):
    features, labels = load_vicon_train(i)
    regr = linear_model.LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    regr.fit(X_train, y_train)
    average_accuracy += mean_squared_error(regr.predict(X_test), y_test)
    models.append(regr)

print(average_accuracy/20)
models.append(DummyModel())
models.append(DummyModel())


frames = load_test_all()
near_models = [DummyModel()]
for i in range(2,22):
    near_models.append(joblib.load(f"models/model_{i}.sav"))

near_models.append(DummyModel())
near_models.append(DummyModel())

predictions = []

counter = 0
for i in range(0, 3412):
    for j in range(0, 23):
        index = i + j*3412
        if near_models[j].predict([frames[index]])[0] == 1.0:
            predictions.append(models[j].predict([frames[index]])[0])
            break
        if j == 22:
            counter += 1
            if len(predictions) != 0:
                predictions.append(predictions[-1])
            else:
                predictions.append([0,0])

print(counter/3412)


csv = {
    'Vicon_x': [i[0] for i in predictions],
    'Vicon_y': [i[1] for i in predictions],
    'Frame_Number': [i for i in range(0, len(predictions))]
}

df = pd.DataFrame(csv, columns=['Frame_Number', 'Vicon_x', 'Vicon_y'])
df.to_csv("vicon_predictions.csv", index = False)


