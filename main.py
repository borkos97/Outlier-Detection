import numpy as np
from numpy import where
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.cof import COF


def prepare_data(filename):
    data = pd.read_csv(filename)
    if filename == 'online_shoppers_intention.csv':
        data = pd.get_dummies(data, columns=['Month', 'VisitorType'])
        data.fillna(0, inplace=True)
    data = StandardScaler().fit_transform(data)
    data = normalize(data)
    data = PCA(n_components=2).fit_transform(data)
    data = pd.DataFrame(data)
    data.columns = ['P1', 'P2']
    return data[:5000]


def CDBSCAN(dataset, eps, min_samples, name):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dataset)
    groups = clustering.labels_
    dataset['COLORS'] = groups
    for group in np.unique(groups):
        label = 'Cluster {}'.format(group) if group != -1 else 'Noise points'
        filtered_group = dataset[dataset['COLORS'] == group]
        plt.scatter(filtered_group['P1'], filtered_group['P2'], label=label)
    n_clusters_ = len(set(groups)) - (1 if -1 in groups else 0)
    n_noise_ = list(groups).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    plt.title(name)
    plt.figtext(0.05, 0.8, 'Eps: {}\nMin_samples: {}\nNumber of clusters: {}\nNoise points: {}'.format(eps, min_samples,
                                                                                                       n_clusters_,
                                                                                                       n_noise_),
                fontsize=15)
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    plt.legend()
    plt.show()


def Kmeans(dataset, nclusters, ninit, maxiter, name):
    clustering = KMeans(n_clusters=nclusters, n_init=ninit, max_iter=maxiter).fit(dataset)
    groups = clustering.labels_
    dataset['COLORS'] = groups
    for group in np.unique(groups):
        label = 'Cluster {}'.format(group) if group != -1 else 'Noise points'
        filtered_group = dataset[dataset['COLORS'] == group]
        plt.scatter(filtered_group['P1'], filtered_group['P2'], label=label)

    plt.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1], c="#000000", s=100)
    plt.title(name)
    plt.figtext(0.05, 0.8, 'N_clusters: {}\nN_init: {}\nMax_iter: {}'.format(nclusters, ninit, maxiter),
                fontsize=15)
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    plt.legend(prop={'size': 15})
    plt.show()


def LOF(dataset, n_neighbors, metric, contamination, name, novelty):
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric=metric, contamination=contamination, novelty=novelty)
    y_pred = clf.fit_predict(dataset)
    outlier_index = where(y_pred == -1)
    outlier_values = dataset.iloc[outlier_index]
    number_of_outlier = len(outlier_values)
    plt.title(name, loc='center', fontsize=20)
    plt.scatter(dataset["P1"], dataset["P2"], color="b", s=65)
    plt.scatter(outlier_values["P1"], outlier_values["P2"], color="r")
    plt.figtext(0.7, 0.91,
                'contamination = {}\nn_neighbors = {} \nnumber of outlier = {}'.format(contamination, n_neighbors,
                                                                                      number_of_outlier), fontsize=9)
    plt.show()


def aCOF(dataset, contamination, n_neighbors, name):
    algo = COF(contamination=contamination, n_neighbors=n_neighbors).fit(dataset)
    outlier_labels = algo.predict(dataset)
    outlier_index = where(outlier_labels == 1)
    outlier_values = dataset.iloc[outlier_index]
    number_of_outlier = len(outlier_values)
    plt.title(name, loc='center', fontsize=20)
    plt.scatter(dataset["P1"], dataset["P2"], color="b", s=65)
    plt.scatter(outlier_values["P1"], outlier_values["P2"], color="r")
    plt.figtext(0.7, 0.91,
                'contamination = {}\nn_neighbors = {} \nnumber of outlier = {}'.format(contamination, n_neighbors,
                                                                                      number_of_outlier), fontsize=9)
    plt.show()

ready_file = prepare_data('online_shoppers_intention.csv')
CDBSCAN(ready_file, 0.1, 20, "Wykres dla DBSCAN")
Kmeans(ready_file, 2, 5, 100, "Wykres dla K-Means")
LOF(ready_file, 2, "manhattan", 0.02, "Wykres LOF", False)
aCOF(ready_file, 0.02, 2, "Wykres COF")
LOF(ready_file, 2, "manhattan", 0.1, "Wykres LOF", False)
aCOF(ready_file, 0.1, 2, "Wykres COF")
LOF(ready_file, 2, "manhattan", 0.002, "Wykres LOF", False)
aCOF(ready_file, 0.002, 2, "Wykres COF")

ready_file = prepare_data('winequality-red.csv')
CDBSCAN(ready_file, 0.1, 20, "Wykres dla DBSCAN")
Kmeans(ready_file, 2, 5, 100, "Wykres dla K-Means")
LOF(ready_file, 2, "manhattan", 0.02, "Wykres LOF", False)
aCOF(ready_file, 0.02, 2, "Wykres COF")
LOF(ready_file, 2, "manhattan", 0.1, "Wykres LOF", False)
aCOF(ready_file, 0.1, 2, "Wykres COF")
LOF(ready_file, 2, "manhattan", 0.002, "Wykres LOF", False)
aCOF(ready_file, 0.002, 2, "Wykres COF")
