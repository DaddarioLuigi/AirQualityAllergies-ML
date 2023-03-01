"""
In questo file applico le modifiche necessarie al dataset. Vengono fatte delle chiamate
a delle funzioni definite nel file dataprocess.py

"""

import pandas as pd
import dataprocess
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

df = pd.read_csv("carbonara.csv", delimiter=',', parse_dates=True, index_col=0)
df2 = pd.read_csv("carbonara_con_oversampling.csv", delimiter=";", parse_dates=True, index_col=0)

df = df[["PM10", "NO2", "NO", "NOX", "IQA"]]
df2 = df2[["PM10", "NO2", "NO", "NOX", "IQA"]]

value = df.values
groups = [0, 1, 2, 3, 4]
i = 1

plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(value[:, group])
    plt.title(df.columns[group], y=0.5, loc='right')
    i += 1

plt.show()

wcss = []
for number_of_clusters in range(1, 11):
    kmeans = cluster.KMeans(n_clusters=number_of_clusters, random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

print(wcss)

ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(ks, wcss);
plt.xlabel("K")
plt.ylabel("wcss")
plt.axvline(3, linestyle='--', color='r')
plt.show()

#
# CLUSTERING
#

# clustering (K-means)
processed_dataframe = dataprocess.MakeClusters(df)
processed_df_oversampling = dataprocess.MakeClusters(df2)


# export del dataframe in un file csv
processed_dataframe.to_csv("clustering.csv", index=False)
processed_df_oversampling.to_csv("clustering_con_oversampling.csv", index=False)

c_zero = processed_dataframe[processed_dataframe["clusters"] == 0]
c_one = processed_dataframe[processed_dataframe["clusters"] == 1]
c_two = processed_dataframe[processed_dataframe["clusters"]==2]
c_three = processed_dataframe[processed_dataframe["clusters"]==3]
c_four = processed_dataframe[processed_dataframe["clusters"]==4]

print(c_zero.shape, c_one.shape, c_two.shape, c_three.shape, c_four.shape, processed_dataframe.shape)

# plot
sns.pairplot(data=processed_dataframe, hue="clusters")
plt.show()

# RELAZIONE TRA PM10 E IQA
sns.relplot(
    data=processed_dataframe,
    x=processed_dataframe["PM10"], y=processed_dataframe["NO"], hue="clusters"
)
plt.show()


corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


