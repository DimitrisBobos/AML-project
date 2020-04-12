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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering 
import scipy.cluster.hierarchy as shc 


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


# boxplot dataset after pca
#plt.figure(6)
#X.boxplot()


# convert df from numpy array to pandas dataframe
df = pd.DataFrame(data = X,
                 columns = ['Dimension 1', 'Dimension 2', 'Dimension 3'])


# remove outliers based on z-score
z = np.abs(stats.zscore(X))
df = df[(z < 3).all(axis = 1)]


# boxplot final dataset
plt.figure(7)
boxplot = df.boxplot()
plt.show()


# set seed to replicate results
seed = 1
np.random.seed(seed)


###############################################################################

# +---------------+
# | agglomerative |
# +---------------+

plt.figure(figsize =(25, 8)) 
plt.title('Visualising the data') 
Dendrogram = shc.dendrogram((shc.linkage(df, method ='ward'))) 
plt.axhline(y=30, color='r', linestyle='--')


nClusts = []
silhouette_list = []
for p in range(2,20):

    clusterer = AgglomerativeClustering(n_clusters=p, linkage='ward')
    
    clusterer.fit(df)
    # The higher (up to 1) the better
    nClusts.append(p)
    
    silhouette_list.append(silhouette_score(df, clusterer.fit_predict(df)))


X = pd.DataFrame({'n_clustsers':nClusts,'scores':silhouette_list})
X = X.set_index('n_clustsers')

print(X)
X.plot()

clusterer = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')

score = silhouette_score(df, clusterer.fit_predict(df))
print(score)


# add cluster attribute to df dataset
df['cluster'] = pd.Series(clusterer.fit_predict(df), index=df.index)


# plot data accoding to the cluster they belong
# -> cluster_0
df_c0 = df.where(df['cluster'] == 0)
df_c0 = df_c0.drop(columns = 'cluster')
df_c0 = df_c0.dropna()
plt.figure(0)
boxplot = df_c0.boxplot()
plt.title('Cluster_0 visualization')
plt.show()

# -> cluster_1
df_c1 = df.where(df['cluster'] == 1)
df_c1 = df_c1.drop(columns = 'cluster')
df_c1 = df_c1.dropna()
plt.figure(1)
boxplot = df_c1.boxplot()
plt.title('Cluster_1 visualization')
plt.show()





###############################################################################

