#######################################################
# PREDIZIONE CON DATI DI CASAMASSIMA
#######################################################
# carico il csv
import math as mt
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
from data import dataprocess
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

from models import iqa_prediction
from models.iqa_prediction import scaler

df2 = pd.read_csv("C:/Users/Luigi Daddario/PycharmProjects/AirQualityPollens/models/casamassima_clustering.csv",
                 delimiter=';', header=0, index_col=0)
values2 = df2.values

# scaling con MinMax
scaler_test = RobustScaler()
_X = df2.loc[:, df2.columns != "clusters"]
X = scaler.fit_transform(_X)
X = pd.DataFrame(X)

reframed = dataprocess.series_to_supervised(X, 1, 1)

# not want to predict
reframed.drop(reframed.columns[[1, 2, 3, 4]], axis=1, inplace=True)

# splitting data
values2 = reframed.values

train_X, test_X, train_y, test_y = train_test_split(values2[:, :-1], values2[:, -1], test_size=0.33, random_state=42,
                                                    shuffle=True)

# reshape in 3D [samples, timesteps, features] ricorda!!
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


# make a prediction
yhat = iqa_prediction.model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]


rmse = mt.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE casamassima: %.3f' % rmse)
