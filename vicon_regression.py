import joblib
import numpy as np
from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

from data_loader import load_vicon_train, load_test_all
from near_classification import computeStrips

class DummyModel:
    def fit(self, X_train, X_test):
        pass
    def predict(self, X):
        return np.zeros(1)
    def predict_proba(self, X):
        return np.array([[1, 0]])


models = [DummyModel()]
average_accuracy = 0
for i in range(2, 22):
    features, labels = load_vicon_train(i)
    regr = ensemble.ExtraTreesRegressor(n_estimators=500, n_jobs=-1)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    regr.fit(X_train, y_train)
    average_accuracy += mean_squared_error(regr.predict(X_test), y_test)
    models.append(regr)

print(f"MSE: {average_accuracy/20}")
models.append(DummyModel())
models.append(DummyModel())


frames = load_test_all()
near_models = []
for i in range(1,24):
    near_models.append(joblib.load(f"models/model_{i}.sav"))

predictions = []
stripsPredictions = computeStrips(near_models, frames)
for i in range(len(stripsPredictions)):
    strips = stripsPredictions[i]
    p = []
    for s in strips:
        index = i + (s-1)*3412 #s its im Bereich [1,23]
        p.append(models[s-1].predict([frames[index]])[0])
    assert len(p) > 0 and len(p) < 3
    if len(p) == 1:
        predictions.append(p[0])
    else:
        xPos = (p[0][0] + p[1][0])/2.0
        yPos = (p[0][1] + p[1][1])/2.0
        predictions.append([xPos, yPos])


csv = {
    'Vicon_x': [i[0] for i in predictions],
    'Vicon_y': [i[1] for i in predictions],
    'Frame_Number': [i for i in range(0, len(predictions))]
}

df = pd.DataFrame(csv, columns=['Frame_Number', 'Vicon_x', 'Vicon_y'])
df.to_csv("vicon_predictions.csv", index = False)


