import joblib
import numpy as np
from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd

from data_loader import load_train, load_vicon_train, load_test_all


class DummyModel:
    def fit(self, X_train, X_test):
        pass
    def predict(self, X):
        return np.zeros(1)


models = [DummyModel()]
for i in range(2, 22):
    features, labels = load_vicon_train(i)
    regr = linear_model.LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    regr.fit(X_train, y_train)
    models.append(regr)

models.append(DummyModel())
models.append(DummyModel())


frames = load_test_all()
near_models = [DummyModel()]
for i in range(2,22):
    near_models.append(joblib.load(f"models/model_{i}.sav"))

near_models.append(DummyModel())
near_models.append(DummyModel())

predictions = []

for i in range(len(frames)):
    strip = int(i / 3412)

    if near_models[strip].predict([frames[i]])[0] == 1.0:
        predictions.append(models[strip].predict([frames[i]])[0])



print(predictions)
print(len(predictions))
csv = {
    'vicon_x': predictions
}

df = pd.DataFrame(csv, columns=['Frame_Number', 'vicon_x', 'vicon_y'])
df.to_csv("vicon_predictions.csv")


