# +----------+
# | preamble |
# +----------+

# import libraries
import numpy as np
import pandas as pd
import seaborn as sn
from IPython.display import display
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors


# load the wholesale customers dataset and drop channel and region
try:
    data = pd.read_csv("Wholesale customers data.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")


# display a description of the dataset
display(data.describe())

# boxplot dataset
plt.figure(1)
data.boxplot()
plt.show()


# produce a correlation matrix for each pair of features in the data
corrMatrix = data.corr()
plt.figure(2)
sn.heatmap(corrMatrix, annot=True, linewidths=.3, cbar=True)


# produce a scatter matrix for each pair of features in the data
plt.figure(3)
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


###############################################################################

# +---------------+
# | preprocessing |
# +---------------+

# scale the data using the natural logarithm
log_data = np.log(data.copy())


# boxplot dataset after scaling
plt.figure(4)
log_data.boxplot()
plt.show()


# investigate variance of components to perform pca
pca = PCA(n_components = 6, svd_solver = 'full')
df = pca.fit_transform(log_data)


# plot explained variance per number of components
plt.figure(5)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
print('Explained variance =', pca.explained_variance_ratio_)
plt.show()


# about 83% variance is achieved for 3 components
# perform pca with 3 principal components
pca = PCA(n_components = 3, svd_solver = 'full')
X = pca.fit_transform(log_data)


# convert df from numpy array to pandas dataframe
df = pd.DataFrame(data = X,
                  columns = ['Dimension 1', 'Dimension 2', 'Dimension 3'])


# boxplot dataset after pca
plt.figure(6)
df.boxplot()
plt.show()


# remove outliers based on z-score
z = np.abs(stats.zscore(X))
df = df[(z < 3).all(axis = 1)]


# boxplot final dataset
plt.figure(7)
df.boxplot()
plt.show()


# set seed to replicate results
seed = 1
np.random.seed(seed)


###############################################################################

# +--------+
# | dbscan |
# +--------+

# determine the best values for epsilon and min_samples for DBSCAN
neigh = NearestNeighbors(n_neighbors=7)
nbrs = neigh.fit(df)
distances, indices = nbrs.kneighbors(df)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
distances = distances[::-1]
plt.figure(8)
plt.plot(distances)
plt.yticks(np.arange(0, 2.3, 0.25))
plt.grid(True)
plt.xlabel('points')
plt.ylabel('k-dist')
plt.title('sorted k-dist graph (k=7)')
plt.show()


# Compute DBSCAN clustering
#db = DBSCAN(eps=0.8, min_samples=7, metric='euclidean').fit(df)
db = DBSCAN(eps=0.8, min_samples=7, metric='euclidean').fit(df)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


# print cluster labels. The value -1 means it's outside all clusters
labels = db.labels_
print (labels)


# evaluate with the silhouette
silhouette_values = silhouette_samples(df, labels)
print ('silhouette=', np.mean(silhouette_values))


# number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print('Silhouette coefficient: %0.3f' % metrics.silhouette_score(df, labels))


# convert cluster labels to dataframe and concat with df
labels = pd.DataFrame(data = labels,
                 columns = ['dbscan_cluster'])
df = pd.concat([df, labels[['dbscan_cluster']]], axis = 1)


# plot data accoding to the cluster they belong

# -> cluster_0
df_c0 = df.where(df['dbscan_cluster'] == 0)
df_c0 = df_c0.drop(columns = 'dbscan_cluster')
df_c0 = df_c0.dropna()
plt.figure(9)
boxplot = df_c0.boxplot()
plt.title('Cluster_0 visualization')
plt.show()


# -> cluster_1
df_c1 = df.where(df['dbscan_cluster'] == 1)
df_c1 = df_c1.drop(columns = 'dbscan_cluster')
df_c1 = df_c1.dropna()
plt.figure(10)
boxplot = df_c1.boxplot()
plt.title('Cluster_1 visualization')
plt.show()


###############################################################################

