import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import sklearn.cluster as cluster
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # time sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def MakeClusters(DataFrame):

    kmeans = cluster.KMeans(n_clusters=5, init="k-means++")

    kmeans = kmeans.fit(DataFrame)

    # aggiungo la colonna al dataset
    DataFrame['clusters'] = kmeans.labels_

    # scatterplot PM10 - IQA
    plot = sns.scatterplot(x=DataFrame["PM10"], y=DataFrame["NOX"], hue=DataFrame['clusters'], data=DataFrame)
    plot.set_xlabel("PM10")
    plot.set_ylabel("NOX")
    fig = plot.get_figure()
    fig.savefig("C:/Users/Luigi Daddario/PycharmProjects/AirQualityPollens/visualization/pm10joiniqa.png")
    plt.show()


    return DataFrame
