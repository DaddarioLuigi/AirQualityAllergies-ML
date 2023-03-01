import pandas as pd
import dataprocess
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster


#######SINTOMI#######
sintomi = pd.read_csv("marzo_sintomi.csv", delimiter=';', parse_dates=True, index_col=0)


df = pd.DataFrame(sintomi)

df = df[["PM10", "IQA", "starnuti", "tosse", "asma", "difficolta_respiratorie", "mal_di_testa"]]

cov_matrix = pd.DataFrame.cov(df)
sns.heatmap(cov_matrix, annot=True, fmt='g')
plt.show()

sintomi=sintomi[["PM10","NO2","NO","NOX","IQA","starnuti","tosse","asma","difficolta_respiratorie","mal_di_testa","sonnolenza","occhi_gonfi","lacrimazione","gola_irritata","prurito"]]

# clustering (K-means)
kmeans = cluster.KMeans(n_clusters=5, init="k-means++")

kmeans = kmeans.fit(sintomi)

# aggiungo la colonna al dataset
sintomi['clusters'] = kmeans.labels_

# scatterplot PM10 - IQA
plot = sns.scatterplot(x=sintomi["IQA"], y=sintomi["starnuti"], hue=sintomi["clusters"], data=sintomi)
plot.set_xlabel("PM10")
plot.set_ylabel("starnuti")
fig = plot.get_figure()
fig.savefig("C:/Users/Luigi Daddario/PycharmProjects/AirQualityPollens/visualization/pm10joinstarnuti.png")
plt.show()

corrMatrix = sintomi[["PM10","IQA","starnuti","tosse","asma","difficolta_respiratorie","mal_di_testa","sonnolenza","occhi_gonfi","lacrimazione","gola_irritata","prurito"]].corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

corrMatrix = sintomi[["NO2","NO","NOX","starnuti","tosse","asma","difficolta_respiratorie","mal_di_testa","sonnolenza","occhi_gonfi","lacrimazione","gola_irritata","prurito"]].corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

sns.displot(sintomi, x="IQA", hue="clusters", kind="kde", fill=True)
plt.show()

